import yaml

from fed_exchange_weight_bias.client import Clients
from fed_exchange_weight_bias.server import Server
from fed_exchange_weight_bias.utils.logger import initialize_logging, create_federated_logger

with open("parameters.yaml", mode='r', encoding="utf-8") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

if __name__ == "__main__":
    dataset = "cars"
    model_name = "vgg19"

    """Set hyper-parameters."""
    epoch = params["fed_epochs"]
    learning_rate = params["learning_rate"]
    CLIENTS_NUMBER = params["clients_num"]
    # And as federated learning is online,
    # participants are uncertain about their online status in each training epoch.
    CLIENT_RATIO_PER_ROUND = params["client_ratio_per_round"]
    # Some characteristics of the dataset cifar-10.
    input_shape = params[dataset]["input_shape"]
    classes_num = params[dataset]["classes_num"]

    """Initialize logger."""
    initialize_logging(filepath="logs/", filename="federated_learning.log")
    federated_logger = create_federated_logger("federated learning")

    """Build clients, server."""
    client = Clients(dataset=dataset,
                     model_name=model_name,
                     input_shape=input_shape,
                     classes_num=classes_num,
                     learning_rate=learning_rate,
                     clients_num=CLIENTS_NUMBER)
    server = Server()

    """Begin training."""
    for ep in range(epoch):
        # Empty local_parameters_sum at the beginning of each epoch.
        server.initialize_local_parameters_sum()
        # Choose a random selection of active clients to train in this epoch.
        active_clients = client.choose_clients(CLIENT_RATIO_PER_ROUND)
        # Train these clients.
        for client_id in active_clients:
            client.current_cid = client_id
            print("[fed-epoch {}] cid: {}".format(ep, client_id))
            federated_logger.info("[fed-epoch {}] cid: {}".format(ep, client_id))
            client.download_global_parameters(server.global_parameters)
            client.train_local_model(train_ratio=params["train_ratio"],
                                     local_epochs=params["local_epochs"],
                                     batch_size=params["batch_size"])
            # Accumulate local parameters.
            current_local_parameters = client.upload_local_parameters()
            server.accumulate_local_parameters(current_local_parameters)
        # Update global parameters in each epoch.
        server.update_global_parameters(len(active_clients))
