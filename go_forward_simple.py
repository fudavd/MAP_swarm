import time
from tdmclient import ClientAsync
global aruco_spotted

start_time = time.time()

targets_g = {"motor.left.target": [int(0)], "motor.right.target": [int(0)]}
aruco_spotted = 0

with ClientAsync() as client:
    
    async def change_node_var():
        with await client.lock() as node:
            await node.set_variables(targets_g)

    def call_program():
        client.run_async_program(change_node_var)

    def aruco_spotted_update(val):
        global aruco_spotted
        aruco_spotted = val

    if aruco_spotted > 0:
        left = 200
        right = 200
        print("go")
    else:
        left = 0
        right = 0
        print("stop")

    targets_g = {"motor.left.target": [int(left)], "motor.right.target": [int(right)]}
    call_program()