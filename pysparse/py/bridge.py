import socket
import os
import json
import atexit

from config import info
from ppo import Args
from runner import Runner

# from pyinstrument import Profiler

def main():
    server_address = './pysparse.sock'

    # Make sure the socket does not already exist
    try:
        os.unlink(server_address)
    except OSError:
        if os.path.exists(server_address):
            raise

    # Create a UDS socket
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    # Bind the socket to the address
    info('Starting server on {}'.format(server_address))
    sock.bind(server_address)

    # Listen for incoming connections
    sock.listen(1)

    runner = None

    @atexit.register
    def cleanup():
        # Clean up the connection
        info("Server shutting down, connection closing!")
        connection.close()
        os.remove("pyready.txt")

    while True:
        # Wait for a connection
        info('Ready, waiting for connection')
        open("pyready.txt", "w").close()
        connection, client_address = sock.accept()
        info('Incoming connection from client', client_address)

        # Receive the data in small chunks
        buffer = ""
        while True:
            data = connection.recv(64).decode('utf-8')
            if not data:
                break
                
            buffer += data
            # info('Received {!r}'.format(data))
            
            # Check if terminator is in buffer
            if ';' in buffer:
                # Split at first semicolon
                message, remainder = buffer.split(';', 1)
                # info('Complete message received: {!r}'.format(message))

                # Parse the message and execute the appropriate command
                cmd, *args = message.split('|')
                # info(f"Command: {cmd}")
                # info(f"Args: {args}")
                match cmd:
                    case "step":
                        tree = json.loads(args[2])
                        itm, prob = runner.step(int(args[0]), tree)
                        connection.sendall(f"{itm}|{prob}".encode('utf-8'))
                    case "next":
                        rewards = json.loads(args[1])
                        terms = json.loads(args[2])
                        fbacks = json.loads(args[3])
                        top_prune = json.loads(args[4])
                        runner.next(int(args[0]), rewards, terms, fbacks, top_prune)
                    case "step_eval":
                        tree = json.loads(args[2])
                        itm, prob = runner.step_eval(int(args[0]), tree)
                        connection.sendall(f"{itm}|{prob}".encode('utf-8'))
                    case "next_eval":
                        rewards = json.loads(args[1])
                        terms = json.loads(args[2])
                        fbacks = json.loads(args[3])
                        runner.next_eval(int(args[0]), rewards, terms, fbacks)
                    case "train":
                        runner.train(int(args[0]))
                    case "save":
                        runner.save()
                    case "load":
                        runner.load(args[0], int(args[1]))
                    case "tmode":
                        runner.t()
                    case "emode":
                        runner.e()
                    case "rs":
                        runner.rs(float(args[0]), int(args[1]), int(args[2]))
                    case "plot":
                        runner.plot(float(args[0]), float(args[1]), float(args[2]), float(args[3]))
                    case "plot_eval":
                        runner.plot(0.0, 0.0, float(args[0]), float(args[1]))
                    case "dir":
                        connection.sendall(f"{runner.args.save_dir}/{runner.run_name}".encode('utf-8'))
                    case "opts":
                        opts = json.loads(args[0])
                        args = Args()
                        args.update_from_dict(opts)
                        runner = Runner(args)
                        runner.init_writer()
                        info("Running with args:", runner.args)
                        runner.writer.add_text(
                            "hyperparameters",
                            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(runner.args).items()])),
                        )

                connection.sendall(";".encode('utf-8'))
                
                # Keep remainder in buffer for next message
                buffer = remainder


if __name__ == "__main__":
    main()