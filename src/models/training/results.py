from matplotlib import pyplot as plt
from torch import Tensor
from dataset.utils import info, warn


def compute_accuracy(outputs: Tensor, labels: Tensor) -> float:
    preds = outputs.argmax(-1)
    acc = (preds == labels.view_as(preds)).float().detach().cpu().numpy().mean()
    return acc


class Results:
    def __init__(self, epochs: int, name:str ="", log_interval: int = 10000):
        self.name = name
        self.max_epochs: int = epochs
        self.losses: list[float] = []
        self.accuracies: list[float] = []
        self._current_losses: list[float] = []
        self._current_accuracies: list[float] = []
        self.log_interval: int = log_interval
        self.last_log: int = 0

    def __str__(self):
        avg_acc = self.accuracies[-1]
        avg_loss = self.losses[-1]
        return f"{self.name}: Acc={avg_acc:.2f} L={avg_loss:.2f} "

    @property
    def current_epoch(self) -> int:
        return len(self.losses) + 1

    def new_epoch(self):
        if len(self._current_losses) > 0:
            last_loss = sum(self._current_losses)/len(self._current_losses)
            last_acc = sum(self._current_accuracies) / len(self._current_accuracies)
            self.losses.append(last_loss)
            self.accuracies.append(last_acc)
            self._current_losses = []
            self._current_accuracies = []
            info(f"{self.name} Avg Ac={last_acc:5.2f}  L={last_loss:5.2f}")
        info(f"{self.name} Epoch {self.current_epoch}/{self.max_epochs}")

    def finish(self):
        self.new_epoch()

    def append(self, loss: float, output, target):
        self._current_losses.append(loss)
        accuracy = compute_accuracy(output, target)
        self._current_accuracies.append(accuracy)
        if self.last_log:
            self.last_log -= 1
        else:
            self.last_log = self.log_interval
            info(f"{self.name} Ac={accuracy:5.2f} L={loss:5.2f}")


def plot_results(results_list: list[Results], title: str = ""):
    plt.figure()
    plt.suptitle(title)

    plt.subplot(2, 1, 1)
    for results in results_list:
        if len(results.accuracies) == 1:
            plt.axhline(y=results.accuracies[0], color='green', label=results.name)
        else:
            plt.plot(results.accuracies, label=results.name)
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()

    plt.subplot(2, 1, 2)
    for results in results_list:
        if len(results.losses) == 1:
            plt.axhline(y=results.losses[0], color='green', label=results.name)
        else:
            plt.plot(results.losses, label=results.name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()


    plt.show()
