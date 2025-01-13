from torchvision.datasets import CIFAR10
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from torchvision import transforms

from time import time

class ArtBench10(CIFAR10):

    base_folder = "artbench-10-batches-py"
    url = "https://artbench.eecs.berkeley.edu/files/artbench-10-python.tar.gz"
    filename = "artbench-10-python.tar.gz"
    tgz_md5 = "9df1e998ee026aae36ec60ca7b44960e"
    train_list = [
        ["data_batch_1", "c2e02a78dcea81fe6fead5f1540e542f"],
        ["data_batch_2", "1102a4dcf41d4dd63e20c10691193448"],
        ["data_batch_3", "177fc43579af15ecc80eb506953ec26f"],
        ["data_batch_4", "566b2a02ccfbafa026fbb2bcec856ff6"],
        ["data_batch_5", "faa6a572469542010a1c8a2a9a7bf436"],
    ]

    test_list = [
        ["test_batch", "fa44530c8b8158467e00899609c19e52"],
    ]
    meta = {
        "filename": "meta",
        "key": "styles",
        "md5": "5bdcafa7398aa6b75d569baaec5cd4aa",
    }

if __name__ == "__main__":
    train_dataset = ArtBench10(root="data", download=True, transform=transforms.ToTensor())
    test_dataset = ArtBench10(root="data", download=True, train=False, transform=transforms.ToTensor())

    X_train, Y_train = train_dataset.data, train_dataset.targets
    X_test, Y_test = test_dataset.data, test_dataset.targets

    # reshape X_train and X_test from (N, 32, 32, 3) to (N, 32*32*3)
    X_train = X_train.reshape(X_train.shape[0], 32*32*3)
    X_test = X_test.reshape(X_test.shape[0], 32*32*3)

    tm = MultiClassTsetlinMachine(4000, 50*100, 5.0, weighted_clauses=True)

    print("\nAccuracy over 30 epochs:\n")
    for i in range(30):
        start = time()
        tm.fit(X_train, Y_train, epochs=1, incremental=True)
        stop = time()
        
        result = 100*(tm.predict(X_test) == Y_test).mean()
        
        print("#%d Accuracy: %.2f%% (%.2fs)" % (i+1, result, stop-start))


