import os.path
from math import ceil
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.utils.data.dataset as dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from archs.puremd_torch.data_container import DataContainer
# from dimenet.data_container import DataContainer


class Init_Dataset:
    def __init__(self, application, write_down=False, delete_constant=False):
        self.application = application
        self.LengthOfDataset = None
        if application == "fluidanimation":
            self.LengthOfDataset = 3294120
            self.input_vector_dir = "/aul/homes/sgao014/Projects/AI4Science/ML-sorrogate/Dataset/fluidanimation/inputvector_test.txt"
            self.input_coef_dir = "/aul/homes/sgao014/Projects/AI4Science/ML-sorrogate/Dataset/fluidanimation/inputcoef_test.txt"
            self.output_vector_dir = (
                "/aul/homes/sgao014/Projects/AI4Science/ML-sorrogate/Dataset/fluidanimation/outputvector_test.txt"
            )
        elif application == "CFD":
            self.LengthOfDataset = 2329104
            self.input_file = "/aul/homes/sgao014/Projects/AI4Science/ML-sorrogate/Dataset/CFD/input.txt"
            self.output_file = "/aul/homes/sgao014/Projects/AI4Science/ML-sorrogate/Dataset/CFD/output.txt"
        elif application == "puremd":
            self.LengthOfDataset = 1040407
        else:
            print("Application error")
        self.prime_X = None  # prime input for out scope (numpy array)
        self.prime_Y = None  # prime output for out scope (numpy array)

        self.get_data()

        # self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
        #     self.prime_X, self.prime_Y, test_size=0.2, random_state=42
        # )
        if write_down:
            self.write_train_test(delete_constant)

    def write_train_test(self, delete_constant=False):
        save_file = None
        X_train = None
        X_test = None
        if self.application == "fluid":
            # for fluid animation application, we need to remove the columns of constants.
            filters = [1, 2, 3, 4, 5]
            save_file = "../Dataset/fluidanimation/"
            X_train = self.X_train[
                :, [col for col in range(0, 20) if col not in filters]
            ]
            X_test = self.X_test[:, [col for col in range(0, 20) if col not in filters]]
        elif self.application == "CFD":
            save_file = "../Dataset/CFD/"
        elif self.application == "puremd":
            save_file = "../Dataset/puremd/"
        else:
            print("Application error")

        # write down the datas X_train, X_test, Y_train, Y_test into specific files
        with open(os.path.join(save_file, "X_train_filtered.txt"), "w") as f:
            for i in range(X_train.shape[0]):
                for j in range(X_train.shape[1]):
                    f.write(str(X_train[i][j].item()) + " ")
                f.write("\n")

        with open(os.path.join(save_file, "X_test_filtered.txt"), "w") as f:
            for i in range(X_test.shape[0]):
                for j in range(X_test.shape[1]):
                    f.write(str(X_test[i][j].item()) + " ")
                f.write("\n")

        with open(os.path.join(save_file, "Y_train.txt"), "w") as f:
            for i in range(self.Y_train.shape[0]):
                for j in range(self.Y_train.shape[1]):
                    f.write(str(self.Y_train[i][j].item()) + " ")
                f.write("\n")

        with open(os.path.join(save_file, "Y_test.txt"), "w") as f:
            for i in range(self.Y_test.shape[0]):
                for j in range(self.Y_test.shape[1]):
                    f.write(str(self.Y_test[i][j].item()) + " ")
                f.write("\n")

    def get_data(self):
        if self.application == "fluidanimation":
            self.get_data_fluid()
        elif self.application == "CFD":
            self.get_data_CFD()
        elif self.application == "puremd":
            self.get_data_puremd()
        else:
            print("Application error")

    def get_data_fluid(self):
        NUM_ATOMS = 3
        NUM_DIMENSIONS = 4
        with open(self.input_vector_dir, "r") as f:
            num_lines = NUM_ATOMS * NUM_DIMENSIONS * self.LengthOfDataset
            x2 = [
                float(value) for _ in range(num_lines) for value in f.readline().split()
            ]
        vector = self.reshape_vector(x2, 3, 4)

        NUM_ATOMS = 8
        NUM_DIMENSIONS = 1
        # Read the program input
        with open(self.input_coef_dir, "r") as f:
            x1 = [
                float(f.readline())
                for _ in range(NUM_ATOMS * NUM_DIMENSIONS * self.LengthOfDataset)
            ]

        # Create X and Y lists using list comprehension
        coef = self.reshape_vector(x1, 8, 1)

        NUM_ATOMS = 3
        NUM_DIMENSIONS = 1
        # Read the program input
        with open(self.output_vector_dir, "r") as f:
            num_lines = NUM_ATOMS * NUM_DIMENSIONS * self.LengthOfDataset
            x5 = [
                float(value) for _ in range(num_lines) for value in f.readline().split()
            ]

        X = [c + v for c, v in zip(coef, vector)]

        X = delete_constant_columns(np.array(X))
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)

        # Create X and Y lists using list comprehension
        self.prime_Y = np.array(self.reshape_vector(x5, 3, 1))
        self.prime_X = np.array(X_normalized)

    def get_data_CFD(self):
        NUM_ATOMS = 35
        NUM_DIMENSIONS = 1
        with open(self.input_file, "r") as f:
            xyz = [
                float(f.readline())
                for _ in range(NUM_ATOMS * NUM_DIMENSIONS * self.LengthOfDataset)
            ]

        # Create X and Y lists using list comprehension
        X = self.reshape_vector(xyz, NUM_ATOMS, NUM_DIMENSIONS)

        with open(self.output_file, "r") as f:
            out = [float(f.readline()) for _ in range(self.LengthOfDataset * 5)]

        NUM_ATOMS = 5
        NUM_DIMENSIONS = 1

        X = delete_constant_columns(np.array(X))
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)

        Y = self.reshape_vector(out, NUM_ATOMS, NUM_DIMENSIONS)

        self.prime_X = np.array(X_normalized)
        self.prime_Y = np.array(Y)

    def get_data_puremd(self):
        NUM_ATOMS = 4
        NUM_DIMENSIONS = 1
        LengthOfDatast = self.LengthOfDataset
        # Read the program input
        with open("/aul/homes/sgao014/Projects/AI4Science/ML-sorrogate/Dataset/puremd/input_inner_loop_data3/C2dbo.txt", "r") as f:
            x1 = [
                float(f.readline())
                for _ in range(NUM_ATOMS * NUM_DIMENSIONS * LengthOfDatast)
            ]

        # Create X and Y lists using list comprehension
        C3dbo = self.reshape_vector(x1, NUM_ATOMS, NUM_DIMENSIONS)

        NUM_ATOMS = 3
        NUM_DIMENSIONS = 1
        # Read the program input
        with open("/aul/homes/sgao014/Projects/AI4Science/ML-sorrogate/Dataset/puremd/input_inner_loop_data3/ext_press.txt", "r") as f:
            num_lines = NUM_ATOMS * NUM_DIMENSIONS * LengthOfDatast
            x2 = [
                float(value) for _ in range(num_lines) for value in f.readline().split()
            ]

        ext_press = self.reshape_vector(x2, NUM_ATOMS, NUM_DIMENSIONS)

        NUM_ATOMS = 2
        NUM_DIMENSIONS = 1
        # Read the program input
        with open("/aul/homes/sgao014/Projects/AI4Science/ML-sorrogate/Dataset/puremd/input_inner_loop_data3/index.txt", "r") as f:
            num_lines = NUM_ATOMS * NUM_DIMENSIONS * LengthOfDatast
            x2 = [
                float(value) for _ in range(num_lines) for value in f.readline().split()
            ]

        index = self.reshape_vector(x2, NUM_ATOMS, NUM_DIMENSIONS)

        NUM_ATOMS = 3
        NUM_DIMENSIONS = 1
        # Read the program input
        with open(
            "/aul/homes/sgao014/Projects/AI4Science/ML-sorrogate/Dataset/puremd/input_inner_loop_data3/nbr_k_bo_data2.txt", "r"
        ) as f:
            num_lines = NUM_ATOMS * NUM_DIMENSIONS * LengthOfDatast
            x3 = [
                float(value) for _ in range(num_lines) for value in f.readline().split()
            ]

        nbr_k = self.reshape_vector(x3, NUM_ATOMS, NUM_DIMENSIONS)

        NUM_ATOMS = 3
        NUM_DIMENSIONS = 1
        # Read the program input
        with open("/aul/homes/sgao014/Projects/AI4Science/ML-sorrogate/Dataset/puremd/input_inner_loop_data3/temp.txt", "r") as f:
            num_lines = NUM_ATOMS * NUM_DIMENSIONS * LengthOfDatast
            x4 = [
                float(value) for _ in range(num_lines) for value in f.readline().split()
            ]

        temp = self.reshape_vector(x4, NUM_ATOMS, NUM_DIMENSIONS)

        NUM_ATOMS = 3
        NUM_DIMENSIONS = 2
        # Read the program input
        with open("/aul/homes/sgao014/Projects/AI4Science/ML-sorrogate/Dataset/puremd/input_inner_loop_data3/rel_box.txt", "r") as f:
            num_lines = NUM_ATOMS * NUM_DIMENSIONS * LengthOfDatast
            x4 = [
                float(value) for _ in range(num_lines) for value in f.readline().split()
            ]

        rel_box = self.reshape_vector(x4, NUM_ATOMS, NUM_DIMENSIONS)

        NUM_ATOMS = 3
        NUM_DIMENSIONS = 1
        # Read the program output
        with open(
            "/aul/homes/sgao014/Projects/AI4Science/ML-sorrogate/Dataset/puremd/input_inner_loop_data3/temp_output_2.txt", "r"
        ) as f:
            num_lines = NUM_ATOMS * NUM_DIMENSIONS * LengthOfDatast
            x4 = [
                float(value) for _ in range(num_lines) for value in f.readline().split()
            ]

        temp_output = self.reshape_vector(x4, NUM_ATOMS, NUM_DIMENSIONS)

        self.prime_X = [
            c2dbo_row + nbr_k_row + temp_row
            for c2dbo_row, nbr_k_row, temp_row in zip(C3dbo, temp, nbr_k)
        ]

        # self.prime_X = [
        #     index_row + c2dbo_row + ext_press_row + nbr_k_row + temp_row + rel_box_row
        #     for index_row, c2dbo_row, ext_press_row, nbr_k_row, temp_row, rel_box_row in zip(
        #         index, C3dbo, ext_press, temp, nbr_k, rel_box
        #     )
        # ]

        # self.prime_X = np.array(self.prime_X)[:-400000, :]
        # self.prime_Y = np.array(temp_output)[:-400000, :]
        self.prime_X = np.array(self.prime_X)
        self.prime_Y = np.array(temp_output)



    def reshape_vector(self, vector, num_atoms, num_dimensions):
        array = [
            [
                vector[(num_atoms * num_dimensions * j) + (k * num_dimensions) + d]
                for k in range(num_atoms)
                for d in range(num_dimensions)
            ]
            for j in range(self.LengthOfDataset)
        ]
        return array


def feature_selectors(x, y, n_components, method):
    pass


class Load_Dataset(Dataset):
    def __init__(
        self,
        x_path=None,
        y_path=None,
        application="fluid",
        filters=None,
        filter_method="PCA",
        device=None,
        normalize=False,
    ):
        self.application = application
        self.mean_Y = None
        self.std_Y = None
        self.prime_Y = None  # data from file (numpy array)
        self.prime_X = None  # data from file (numpy array)
        self.indices = None  # indices of the selected features
        self.x = torch.tensor([])  # input of dataset for dataloader
        self.y = torch.tensor([])  # output of dataset for dataloader
        self.get_data_from_file(x_path, y_path)  # initial prime_X and prime_Y
        # standardize prime_X and prime_Y, and assign them to x and y
        self.set_mean_std()
        if normalize:
            self.standardize_data()
        else:
            self.x = torch.tensor(self.prime_X, dtype=torch.float32)
            self.y = torch.tensor(self.prime_Y, dtype=torch.float32)
        # self.filter_data(filters, filter_method)
        if device is not None:
            self.x = self.x.to(device)
            self.y = self.y.to(device)
        # if application=='puremd':  # fixme: seems not correct
        #     self.x = self.x[:-400000, :]
        #     self.y = self.y[:-400000, :]

    def standardize_data(self):
        # # normalize the data between 0 and 1 along dimension 0
        # scaler = StandardScaler()
        # X_standardized = scaler.fit_transform(self.prime_X)
        # Y_standardized = (self.prime_Y - np.array(self.mean_Y)) / np.array(self.std_Y)
        X_standardized = self.prime_X
        Y_standardized = self.prime_Y
        # X_normalized = (X_standardized - np.min(X_standardized, axis=0) - 1e-6) / (
        #     np.max(X_standardized, axis=0) - np.min(X_standardized, axis=0) + 2e-6
        # )
        # Y_normalized = (Y_standardized - np.min(Y_standardized, axis=0) - 1e-6) / (
        #     np.max(Y_standardized, axis=0) - np.min(Y_standardized, axis=0) + 2e-6
        # )
        scaler = MinMaxScaler(feature_range=(0.1, 0.9))
        X_normalized = scaler.fit_transform(X_standardized)
        Y_normalized = scaler.fit_transform(Y_standardized)

        self.x = torch.tensor(X_normalized, dtype=torch.float32)
        self.y = torch.tensor(Y_normalized, dtype=torch.float32)

    def get_data_from_file(self, x_path, y_path):
        # Read data from file in file_path, each line is a vector
        if isinstance(x_path, str):
            with open(x_path, "r") as f:
                data_lines = f.readlines()
                x = [[float(value) for value in line.split()] for line in data_lines]
            with open(y_path, "r") as f:
                data_lines = f.readlines()
                y = [[float(value) for value in line.split()] for line in data_lines]
            self.prime_X = np.array(x)
            self.prime_Y = np.array(y)
        elif isinstance(x_path, list):
            # if there is multiple files, we need to read them one by one
            for i in range(len(x_path)):
                with open(x_path[i], "r") as f:
                    data_lines = f.readlines()
                    x = [[float(value) for value in line.split()] for line in data_lines]
                with open(y_path[i], "r") as f:
                    data_lines = f.readlines()
                    y = [[float(value) for value in line.split()] for line in data_lines]
                if i == 0:
                    self.prime_X = np.array(x)
                    self.prime_Y = np.array(y)
                else:
                    self.prime_X = np.concatenate((self.prime_X, np.array(x)), axis=0)
                    self.prime_Y = np.concatenate((self.prime_Y, np.array(y)), axis=0)

    def filter_data(self, filters, filter_method):
        if filters is not None:
            if isinstance(filters, list):
                self.x = self.x[:, filters]
            elif isinstance(filters, int) or isinstance(filters, feature_selectors):
                fs = feature_selectors(
                    self.x, self.y, n_components=filters, method=filter_method
                )
                self.x = torch.tensor(fs.fit(), dtype=torch.float32)
                self.indices = fs.indices
            else:
                print("filters type error")

    def set_mean_std(self):
        """Set the mean and std of the dataset, used for standardization and recover"""
        if self.application == "fluidanimation":
            self.mean_Y = [0.12768913, 0.05470352, 0.14003364]
            self.std_Y = [2.07563408, 1.59399168, 2.06319435]
        elif self.application == "CFD":
            self.mean_Y = [
                -2.01850154e-08,
                -5.23806580e-11,
                7.29894413e-12,
                5.15219587e-12,
                1.15924407e-11,
            ]
            self.std_Y = [0.38392921, 0.12564681, 0.12619844, 0.21385977, 0.68862844]
        elif self.application == "puremd":
            self.mean_Y = [
                1.8506536384110004e-06,
                -0.003247667874206878,
                0.0007951518742184539,
            ]
            self.std_Y = [0.30559314964628986, 0.44421521232555966, 0.4909015024281119]
        else:
            print("Application error")

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class Load_Dataset_or(Dataset):
    def __init__(
        self,
        x_path=None,
        y_path=None,
        application="fluid",
        filters=None,
        filter_method="PCA",
        device=None,
        normalize=False,
    ):
        self.or_data = Init_Dataset(application)
        self.prime_X = self.or_data.prime_X
        self.prime_Y = self.or_data.prime_Y
        self.application = application
        self.mean_Y = None
        self.std_Y = None
        self.indices = None  # indices of the selected features
        self.x = torch.tensor(self.prime_X, dtype=torch.float32)  # input of dataset for dataloader
        self.y = torch.tensor(self.prime_Y, dtype=torch.float32)  # output of dataset for dataloader
        # standardize prime_X and prime_Y, and assign them to x and y
        self.set_mean_std()
        if normalize:
            self.standardize_data()
        # self.filter_data(filters, filter_method)
        if device is not None:
            self.x = self.x.to(device)
            self.y = self.y.to(device)
        # if application=='puremd':  # fixme: seems not correct
        #     self.x = self.x[:-400000, :]
        #     self.y = self.y[:-400000, :]

    def standardize_data(self):
        # # normalize the data between 0 and 1 along dimension 0
        # scaler = StandardScaler()
        # X_standardized = scaler.fit_transform(self.prime_X)
        # Y_standardized = (self.prime_Y - np.array(self.mean_Y)) / np.array(self.std_Y)
        X_standardized = self.prime_X
        Y_standardized = self.prime_Y
        # X_normalized = (X_standardized - np.min(X_standardized, axis=0) - 1e-6) / (
        #     np.max(X_standardized, axis=0) - np.min(X_standardized, axis=0) + 2e-6
        # )
        # Y_normalized = (Y_standardized - np.min(Y_standardized, axis=0) - 1e-6) / (
        #     np.max(Y_standardized, axis=0) - np.min(Y_standardized, axis=0) + 2e-6
        # )
        scaler = MinMaxScaler(feature_range=(0.1, 0.9))
        X_normalized = scaler.fit_transform(X_standardized)
        Y_normalized = scaler.fit_transform(Y_standardized)

        self.x = torch.tensor(X_normalized, dtype=torch.float32)
        self.y = torch.tensor(Y_normalized, dtype=torch.float32)

    def get_data_from_file(self, x_path, y_path):
        # Read data from file in file_path, each line is a vector
        if isinstance(x_path, str):
            with open(x_path, "r") as f:
                data_lines = f.readlines()
                x = [[float(value) for value in line.split()] for line in data_lines]
            with open(y_path, "r") as f:
                data_lines = f.readlines()
                y = [[float(value) for value in line.split()] for line in data_lines]
            self.prime_X = np.array(x)
            self.prime_Y = np.array(y)
        elif isinstance(x_path, list):
            # if there is multiple files, we need to read them one by one
            for i in range(len(x_path)):
                with open(x_path[i], "r") as f:
                    data_lines = f.readlines()
                    x = [[float(value) for value in line.split()] for line in data_lines]
                with open(y_path[i], "r") as f:
                    data_lines = f.readlines()
                    y = [[float(value) for value in line.split()] for line in data_lines]
                if i == 0:
                    self.prime_X = np.array(x)
                    self.prime_Y = np.array(y)
                else:
                    self.prime_X = np.concatenate((self.prime_X, np.array(x)), axis=0)
                    self.prime_Y = np.concatenate((self.prime_Y, np.array(y)), axis=0)

    def filter_data(self, filters, filter_method):
        if filters is not None:
            if isinstance(filters, list):
                self.x = self.x[:, filters]
            elif isinstance(filters, int) or isinstance(filters, feature_selectors):
                fs = feature_selectors(
                    self.x, self.y, n_components=filters, method=filter_method
                )
                self.x = torch.tensor(fs.fit(), dtype=torch.float32)
                self.indices = fs.indices
            else:
                print("filters type error")

    def set_mean_std(self):
        """Set the mean and std of the dataset, used for standardization and recover"""
        if self.application == "fluidanimation":
            self.mean_Y = [0.12768913, 0.05470352, 0.14003364]
            self.std_Y = [2.07563408, 1.59399168, 2.06319435]
        elif self.application == "CFD":
            self.mean_Y = [
                -2.01850154e-08,
                -5.23806580e-11,
                7.29894413e-12,
                5.15219587e-12,
                1.15924407e-11,
            ]
            self.std_Y = [0.38392921, 0.12564681, 0.12619844, 0.21385977, 0.68862844]
        elif self.application == "puremd":
            self.mean_Y = [
                1.8506536384110004e-06,
                -0.003247667874206878,
                0.0007951518742184539,
            ]
            self.std_Y = [0.30559314964628986, 0.44421521232555966, 0.4909015024281119]
        else:
            print("Application error")

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]



class Load_Dataset_Cosmoflow(Dataset):
    def __init__(self, path, device=None):
        self.tensor_x, self.tensor_y = load_ds_from_dir_single(path)
        self.tensor_x = self.tensor_x
        self.tensor_y = self.tensor_y

    def __getitem__(self, index):
        return self.tensor_x[index], self.tensor_y[index]

    def __len__(self):
        return len(self.tensor_x)

    def analyse_data(self):
        print(
            f"mean: {np.mean([torch.mean(x) for x in self.tensor_x])}, std: {np.std([torch.std(x) for x in self.tensor_x])}"
        )
        print(
            f"max: {np.max([torch.max(x) for x in self.tensor_x])}, min: {np.min([torch.min(x) for x in self.tensor_x])}"
        )
        print(
            f"mean: {np.mean([torch.mean(x) for x in self.tensor_y])}, std: {np.std([torch.std(x) for x in self.tensor_y])}"
        )
        print(
            f"max: {np.max([torch.max(x) for x in self.tensor_y])}, min: {np.min([torch.min(x) for x in self.tensor_y])}"
        )

        np_x = np.stack([x.numpy() for x in self.tensor_x])

        # Count the elements of a high-dimensional array np_x and draw a histogram
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(
            np_x.flatten(),
            bins=100,
            color="skyblue",
            edgecolor="black",
            range=(0.02, 1),
        )
        # 设置标题和标签
        plt.title("Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()


class Visualize_Dataset:
    def __init__(self, data):
        self.data = data
        self.indexes_3d_scatter = [
            [0, 6, 7],
            [8, 9, 10],
            [11, 12, 13],
            [14, 15, 16],
            [17, 18, 19],
        ]

    def visualize_3d_scatter(self):
        for index in self.indexes_3d_scatter:
            # create a figure object
            fig = plt.figure()

            # create a 3d axes
            ax = fig.add_subplot(111, projection="3d")

            # using scatter method to draw 3D scatter plot
            ax.scatter(
                self.data[index[0], :],
                self.data[index[1], :],
                self.data[index[2], :],
                c="r",
                marker="o",
                label="Data Points",
            )

            # set axis labels
            ax.set_xlabel("X Label")
            ax.set_ylabel("Y Label")
            ax.set_zlabel("Z Label")

            plt.show()

    def visualize_histogram(self):
        if self.data.shape[0] > self.data.shape[1]:
            self.data = self.data.T
        for i in range(self.data.shape[0]):
            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.hist(
                self.data[i, :],
                bins=100,
                color="skyblue",
                edgecolor="black",
                # range=(-1, 1),
            )
            # 设置标题和标签
            plt.title("Histogram")
            plt.xlabel("Value")
            plt.ylabel("Frequency")

            # 显示直方图
            # plt.show()
            plt.savefig(f"temp_out/output_{i}.png")

    def correlation_analysis(self):
        # remove the constant column
        r_data = self.data[[0] + [i for i in range(6, 20)], :]
        correlation_matrix = [
            ["0.0000" for _ in range(r_data.shape[0])] for _ in range(r_data.shape[0])
        ]
        for i in range(r_data.shape[0]):
            for j in range(0, i + 1):
                correlation_mat = np.corrcoef(r_data[i, :], r_data[j, :])

                # correlation matrix is a symmetric matrix, so the correlation coefficient is located at (0,1) and (1,0)
                pearson_correlation = correlation_mat[0, 1]
                correlation_matrix[i][j] = "{:.4f}".format(pearson_correlation)
        print("Pearson相关系数:")
        print(correlation_matrix)


def load_data(args, fs_indices=None, normalize=False):
    # filtered means there is no constant columns
    data_dir = '/aul/homes/sgao014/Projects/AI4Science/Lottery-Ticket-Hypothesis-in-Pytorch/Dataset/CFD/'
    data_dir = data_dir.replace("CFD", args.dataset)
    x_train_set_ = data_dir + "X_train_filtered.txt"
    y_train_set_ = data_dir + "Y_train.txt"
    x_test_set_ = data_dir + "X_test_filtered.txt"
    y_test_set_ = data_dir + "Y_test.txt"

    if args.dataset == "cosmoflow":
        train_set = Load_Dataset_Cosmoflow(
            "/aul/homes/sgao014/datasets/cosmoflow/train", args.device
        )
        test_set = Load_Dataset_Cosmoflow(
            "/aul/homes/sgao014/datasets/cosmoflow/validation", args.device
        )
        # merge train and test set
        data_set = dataset.ConcatDataset([train_set, test_set])

        # data_set = Load_Dataset_Cosmoflow(
        #     "/aul/homes/sgao014/datasets/cosmoflow/train", args.device
        # )
        # split train and test set
        train_set, test_set = dataset.random_split(data_set, [1984, 64],
                                                   generator=torch.Generator().manual_seed(42))
    elif args.dataset == 'dimenet':
        data_container = DataContainer(
            '/aul/homes/sgao014/Projects/AI4Science/dimenet/data/qm9_eV.npz',
            cutoff=5.0,
            target_keys=['U0'],
            batch_size=256,
            device=args.device
        )
        train_set, test_set = dataset.random_split(
            data_container, [ceil(0.8 * len(data_container)), len(data_container) - ceil(0.8 * len(data_container))]
        )

        return train_set, test_set
    else:
        data_set = Load_Dataset_or(
            x_path=[x_train_set_, x_test_set_],
            y_path=[y_train_set_, y_test_set_],
            filters=fs_indices,
            application=args.dataset,
            normalize=normalize,
        )
        train_set_len = int(0.8 * len(data_set))
        train_set, test_set = dataset.random_split(data_set, [train_set_len, len(data_set)-train_set_len],
                                                   generator=torch.Generator().manual_seed(42))
    # else:
    #     train_set = Load_Dataset(
    #         x_path=x_train_set_,
    #         y_path=y_train_set_,
    #         filters=fs_indices,
    #         application=args.dataset,
    #         normalize=False,
    #     )
    #     if args.feature_selector == "SFS" and fs_indices is not None:
    #         fs_indices = list(train_set.indices)
    #     test_set = Load_Dataset(
    #         x_path=x_test_set_,
    #         y_path=y_test_set_,
    #         filters=fs_indices,
    #         application=args.dataset,
    #         normalize=False,
    #     )
    return train_set, test_set


# TODO: try to solve the problem of the dataset is unbalanced
def delete_constant_columns(np_array):
    # find out constant columns of X_train
    is_constant = np.all(np_array == np_array[0, :], axis=0)
    # get the indices of the constant columns
    cc = np.where(is_constant == True)
    # get rid of the constant columns in X_train and X_test
    np_array = np.delete(np_array, cc, axis=1)
    return np_array


def data_visualize():
    datas = Init_Dataset("puremd")
    print(np.mean(datas.prime_Y, axis=0))
    print(np.std(datas.prime_Y, axis=0))
    visualizer = Visualize_Dataset(datas.prime_X.T)
    visualizer.visualize_histogram()


def load_ds_from_dir_single(path):
    tensor_x = []
    tensor_y = []
    data_file = [name for name in os.listdir(path) if name.endswith("data.npy")]
    for name_file in data_file:
        path_file = os.path.join(path, name_file)
        x = np.load(path_file, allow_pickle=True)
        x = x.reshape(128, 128, 128, -1)
        x = np.rollaxis(x, 3, 0)
        x = x.astype(np.float32) / 500
        y = np.load(path_file.replace("data.npy", "label.npy"), allow_pickle=True)
        y = y.astype(np.float32) / 2 + 0.5
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        tensor_x.append(x)
        tensor_y.append(y)
        # print("in shape:", x.shape)
        # print("in max:", x.max())
        # print("y", y)

    # tensor_x = torch.stack(tensor_x)
    # tensor_y = torch.stack(tensor_y)
    # dataset = TensorDataset(tensor_x, tensor_y)
    return tensor_x, tensor_y


def get_data_loaders(args_):
    args_.data_dir = args_.data_dir.replace("fluidanimation", args_.application)
    nomorlized = True

    # fs_indices can be an int or list. if it is an int, it means the number of selected features
    # if it is a list, it means the indices of selected features
    if args_.application == 'dimenetpp':
        data_container = DataContainer(
            './dimenet/qm9_eV.npz',
            cutoff=5.0,
            target_keys=['U0'],
            batch_size=args_.batch_size,
            device=args_.device
        )
        train_set, test_set = dataset.random_split(
            data_container, [ceil(0.8*len(data_container)), len(data_container)-ceil(0.8*len(data_container))]
        )
        train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    elif nomorlized:
        train_set, test_set, fs_indices_ = load_data(args_, None)
        test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)
        train_loader = DataLoader(train_set, batch_size=args_.batch_size, shuffle=True)
    else:
        fs_indices_ = None
        train_set, test_set, fs_indices_ = load_data(args_, fs_indices_)
        test_loader = DataLoader(test_set, batch_size=args_.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        train_loader = DataLoader(train_set, batch_size=args_.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    return train_loader, test_loader


def cifar10(augment=True, batch_size=128):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    logging = 'Using'
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        logging += ' augmented'
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    print(logging + ' CIFAR 10.')
    kwargs = {'num_workers': 4, 'pin_memory': torch.cuda.is_available()}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                         transform=transform_train),
        batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transform_test),
        batch_size=batch_size, shuffle=True, **kwargs)
    num_classes = 10

    return train_loader, val_loader, num_classes


def mnist(batch_size=100, pm=False):
    transf = [transforms.ToTensor()]
    def flatten(x):
        return x.view(-1, 784)

    if pm:
        transf.append(transforms.Lambda(flatten))
    transform_data = transforms.Compose(transf)

    kwargs = {'num_workers': 4, 'pin_memory': torch.cuda.is_available()}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transform_data),
        batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transform_data),
        batch_size=batch_size, shuffle=True, **kwargs)
    num_classes = 10

    return train_loader, val_loader, num_classes
