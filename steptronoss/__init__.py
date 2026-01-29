def debug(rank: int | list[int] = 0, stop: bool = False, any_rank: bool = False):
    """Start a debugger, connect to this debugger from VSCode, your launch.json should be like below:

    {
        "version": "0.2.0",
        "inputs": [
            {
                "type": "promptString",
                "id": "WorkerIP",
                "description": "IP of your worker",
                "default": "",
            },
            {
                "type": "promptString",
                "id": "WorkerPort",
                "description": "IP of your worker",
                "default": "5678",
            }
        ],
        "configurations": [
            {
                "name": "Python: 远程连接-Steptron",
                "type": "debugpy",
                "request": "attach",
                "connect": {
                    "host": "${input:WorkerIP}",
                    "port": "${input:WorkerPort}"
                },
                "pathMappings": [
                    {
                        "localRoot": "${workspaceFolder}",
                        "remoteRoot": "${workspaceFolder}"
                    }
                ],
                "justMyCode": false,
            }
        ]
    }
    """
    import socket

    import debugpy
    import torch.distributed as dist

    if not isinstance(rank, list):
        rank = [rank]
    if debugpy.is_client_connected():
        return
    if dist.is_initialized():
        my_rank = dist.get_rank()
    else:
        print("Torch not initialized!")
        my_rank = 0
    if my_rank in rank or any_rank:
        try:
            host = socket.gethostbyname(socket.getfqdn(socket.gethostname()))
            if any_rank:
                port = 5000 + my_rank
            else:
                port = 5000 + rank.index(my_rank)

            debugpy.listen((host, port), in_process_debug_adapter=False)
            print("Waiting for debugger...\a")
            print(f"ip: {host} port:{port} rank: {my_rank}", flush=True)
            debugpy.wait_for_client()
            print("Connected.")
            if stop:
                debugpy.breakpoint()
        except Exception as e:
            import traceback

            print("\n".join(traceback.format_exception(e)))
