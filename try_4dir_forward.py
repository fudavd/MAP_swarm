from tdmclient import ClientAsync
import numpy as np
from fourdir_sensor import FourDirSensor

targets_g = {"motor.left.target": [int(0)], "motor.right.target": [int(0)]}
test = FourDirSensor(verbose=True)
left = 0
right = 0

if __name__ == "__main__":
    test.start()
    with ClientAsync() as client:
        async def change_node_var():
            with await client.lock() as node:
                await node.set_variables(targets_g)

        def call_program():
            client.run_async_program(change_node_var)

        while True:
            left = 0
            right = 0
            output = test.call_buffer()
            if np.any(output[:,0]!=0.0):
                left = 0
                right = 0

            targets_g = {"motor.left.target": [int(left)], "motor.right.target": [int(right)]}
            call_program()


            print(output)

    test.exit()
