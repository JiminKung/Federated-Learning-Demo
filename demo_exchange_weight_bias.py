from fed_exchange_weight_bias.client import Clients
from fed_exchange_weight_bias.server import Server
from fed_exchange_weight_bias.utils.logger import initialize_logging, create_federated_logger

if __name__ == "__main__":
    """Set hyper-parameters."""
    epoch = 6
    learning_rate = 0.0001
    CLIENT_NUMBER = 4
    # And as federated learning is online,
    # participants are uncertain about their online status in each training epoch.
    CLIENT_RATIO_PER_ROUND = 1.00
    # Some characteristics of the dataset cifar-10.
    input_shape = (32, 32, 3)
    classes_num = 10  # cifar-10

    """initialize logger"""
    initialize_logging(filepath="logs/", filename="federated_learning.log")
    federated_logger = create_federated_logger("federated learning")

    """Build clients, server."""
    client = Clients(input_shape=input_shape,
                     classes_num=classes_num,
                     learning_rate=learning_rate,
                     clients_num=CLIENT_NUMBER)
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
            client.train_local_model()
            # Accumulate local parameters.
            current_local_parameters = client.upload_local_parameters()
            server.accumulate_local_parameters(current_local_parameters)
        # Update global parameters in each epoch.
        server.update_global_parameters(len(active_clients))
