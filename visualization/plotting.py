import matplotlib.pyplot as plt

class Visualization:
    def plotting_metric(self, hist_dict, train_metric, val_metric, path, fig_name):
        plt.plot(hist_dict[train_metric], label='Train Cosine_similarity')
        plt.plot(hist_dict[val_metric], label='Validation Cosine_similarity')
        #plt.title('Contextual-Based Model: Cosine Training History')
        plt.ylabel('Cosine Similarity')
        plt.xlabel('Epochs')
        plt.legend(loc="upper left")


        plt.savefig(f'{path}/{fig_name}_learning.pdf', format='pdf')
        # plt.show()
        plt.close()

    def plotting_loss(self, hist_dict, train_loss, val_loss, path, fig_name):
        plt.plot(hist_dict[train_loss], label='Training Loss')
        plt.plot(hist_dict[val_loss], label='Validation Loss')
        #plt.title('Contextual-Based Model: Loss Training History')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(loc="upper left")


        plt.savefig(f'{path}/{fig_name}_loss.pdf', format='pdf')

        # plt.show()
        plt.close()
