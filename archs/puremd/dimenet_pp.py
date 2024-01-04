import ast
import tensorflow as tf
import yaml
from .layers.embedding_block import EmbeddingBlock
from .layers.bessel_basis_layer import BesselBasisLayer
from .layers.spherical_basis_layer import SphericalBasisLayer
from .layers.interaction_pp_block import InteractionPPBlock
from .layers.output_pp_block import OutputPPBlock
from .activations import swish
from .training.trainer import Trainer

with open('./archs/puremd/config_pp.yaml', 'r') as c:
    config = yaml.safe_load(c)

for key, val in config.items():
    if type(val) is str:
        try:
            config[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass

model_name = config['model_name']

if model_name == "dimenet":
    num_bilinear = config['num_bilinear']
elif model_name == "dimenet++":
    out_emb_size = config['out_emb_size']
    int_emb_size = config['int_emb_size']
    basis_emb_size = config['basis_emb_size']
    extensive = config['extensive']
else:
    raise ValueError(f"Unknown model name: '{model_name}'")

emb_size = config['emb_size']
num_blocks = config['num_blocks']

num_spherical = config['num_spherical']
num_radial = config['num_radial']
output_init = config['output_init']

cutoff = config['cutoff']
envelope_exponent = config['envelope_exponent']

num_before_skip = config['num_before_skip']
num_after_skip = config['num_after_skip']
num_dense_output = config['num_dense_output']

num_train = config['num_train']
num_valid = config['num_valid']
data_seed = config['data_seed']
dataset = config['dataset']
logdir = config['logdir']

num_steps = config['num_steps']
ema_decay = config['ema_decay']

learning_rate = config['learning_rate']
warmup_steps = config['warmup_steps']
decay_rate = config['decay_rate']
decay_steps = config['decay_steps']

batch_size = config['batch_size']
evaluation_interval = config['evaluation_interval']
save_interval = config['save_interval']
restart = config['restart']
comment = config['comment']
targets = config['targets']


class DimeNetPP(tf.keras.Model):
    """
    DimeNet++ model.

    Parameters
    ----------
    emb_size
        Embedding size used for the messages
    out_emb_size
        Embedding size used for atoms in the output block
    int_emb_size
        Embedding size used for interaction triplets
    basis_emb_size
        Embedding size used inside the basis transformation
    num_blocks
        Number of building blocks to be stacked
    num_spherical
        Number of spherical harmonics
    num_radial
        Number of radial basis functions
    envelope_exponent
        Shape of the smooth cutoff
    cutoff
        Cutoff distance for interatomic interactions
    num_before_skip
        Number of residual layers in interaction block before skip connection
    num_after_skip
        Number of residual layers in interaction block after skip connection
    num_dense_output
        Number of dense layers for the output blocks
    num_targets
        Number of targets to predict
    activation
        Activation function
    extensive
        Whether the output should be extensive (proportional to the number of atoms)
    output_init
        Initialization method for the output layer (last layer in output block)
    """

    def __init__(
            self,
            emb_size=emb_size,
            out_emb_size=out_emb_size,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            num_blocks=num_blocks,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_dense_output=num_dense_output,
            num_targets=len(targets),
            activation=swish,
            extensive=extensive,
            output_init=output_init,
            name='dimenet',
            **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_blocks = num_blocks
        self.extensive = extensive

        # Cosine basis function expansion layer
        self.rbf_layer = BesselBasisLayer(
            num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)
        self.sbf_layer = SphericalBasisLayer(
            num_spherical, num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)

        # Embedding and first output block
        self.output_blocks = []
        self.emb_block = EmbeddingBlock(emb_size, activation=activation)
        self.output_blocks.append(
            OutputPPBlock(emb_size, out_emb_size, num_dense_output, num_targets,
                          activation=activation, output_init=output_init))

        # Interaction and remaining output blocks
        self.int_blocks = []
        for i in range(num_blocks):
            self.int_blocks.append(
                InteractionPPBlock(emb_size, int_emb_size, basis_emb_size, num_before_skip,
                                   num_after_skip, activation=activation))
            self.output_blocks.append(
                OutputPPBlock(emb_size, out_emb_size, num_dense_output, num_targets,
                              activation=activation, output_init=output_init))

        self.dense_layers = []
        for block in self.int_blocks:
            self.dense_layers.append(block.down_projection)

    def calculate_interatomic_distances(self, R, idx_i, idx_j):
        Ri = tf.gather(R, idx_i)
        Rj = tf.gather(R, idx_j)
        # ReLU prevents negative numbers in sqrt
        Dij = tf.sqrt(tf.nn.relu(tf.reduce_sum((Ri - Rj)**2, -1)))
        return Dij

    def calculate_neighbor_angles(self, R, id3_i, id3_j, id3_k):
        """Calculate angles for neighboring atom triplets"""
        Ri = tf.gather(R, id3_i)
        Rj = tf.gather(R, id3_j)
        Rk = tf.gather(R, id3_k)
        R1 = Rj - Ri
        R2 = Rk - Rj
        x = tf.reduce_sum(R1 * R2, axis=-1)
        y = tf.linalg.cross(R1, R2)
        y = tf.norm(y, axis=-1)
        angle = tf.math.atan2(y, x)
        return angle

    def call(self, inputs):
        Z, R                         = inputs['Z'], inputs['R']
        batch_seg                    = inputs['batch_seg']
        idnb_i, idnb_j               = inputs['idnb_i'], inputs['idnb_j']
        id_expand_kj, id_reduce_ji   = inputs['id_expand_kj'], inputs['id_reduce_ji']
        id3dnb_i, id3dnb_j, id3dnb_k = inputs['id3dnb_i'], inputs['id3dnb_j'], inputs['id3dnb_k']
        n_atoms = tf.shape(Z)[0]

        # Calculate distances
        Dij = self.calculate_interatomic_distances(R, idnb_i, idnb_j)
        rbf = self.rbf_layer(Dij)

        # Calculate angles
        Anglesijk = self.calculate_neighbor_angles(
            R, id3dnb_i, id3dnb_j, id3dnb_k)
        sbf = self.sbf_layer([Dij, Anglesijk, id_expand_kj])

        # Embedding block
        x = self.emb_block([Z, rbf, idnb_i, idnb_j])
        P = self.output_blocks[0]([x, rbf, idnb_i, n_atoms])

        # Interaction blocks
        for i in range(self.num_blocks):
            x = self.int_blocks[i]([x, rbf, sbf, id_expand_kj, id_reduce_ji])
            P += self.output_blocks[i+1]([x, rbf, idnb_i, n_atoms])

        if self.extensive:
            P = tf.math.segment_sum(P, batch_seg)
        else:
            P = tf.math.segment_mean(P, batch_seg)
        return P


def get_trainer(model):
    trainer = Trainer(model, learning_rate, warmup_steps,
                      decay_steps, decay_rate,
                      ema_decay=ema_decay, max_grad_norm=1000)
    return trainer


def extract_dense_layers_from_custom_block(block, dense_layers):
    """Extract Dense layers from a custom block."""
    for attr in dir(block):
        # Skip 'input' and 'output' attributes
        if attr in ['input', 'output']:
            continue
        # Get the actual value of the attribute
        potential_layer = getattr(block, attr)
        # Check if this attribute is a layer
        if isinstance(potential_layer, tf.keras.layers.Layer):
            # Check if this layer is a Dense layer
            if isinstance(potential_layer, tf.keras.layers.Dense):
                dense_layers.append(potential_layer)


def extract_dense_layers(layer, dense_layers):
    """递归函数来提取Dense层"""
    if isinstance(layer, tf.keras.layers.Dense):
        dense_layers.append(layer)
    elif hasattr(layer, 'layers'):
        for sublayer in layer.layers:
            extract_dense_layers(sublayer, dense_layers)
    else:
        # 处理自定义层
        extract_dense_layers_from_custom_block(layer, dense_layers)

