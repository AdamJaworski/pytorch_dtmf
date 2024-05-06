import torch.nn as nn
import utilities


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target):
        output_code = utilities.tensor_to_code(output)
        print(f'Output code: {output_code}  Expected: {target}')
        string1 = output_code.split()
        string2 = target.split()

        A = set(string1)
        B = set(string2)

        str_diff = A.symmetric_difference(B)
        isEmpty = (len(str_diff) == 0)

        if isEmpty:
            return 0

        return len(str_diff)
