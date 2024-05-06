import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch_directml
import torch.optim as optim
import utilities
from dataset import Dataset
import torch.nn as nn
from model import DTMFNet
from training_options import opt
from PathManager import PathManager
import pathlib
from datetime import datetime


wav_data = r'./Data/verified/'


def train_model(model: DTMFNet, loss_fn):
    dml = torch_directml.device()
    model.to(dml, non_blocking=True)
    optimizer         = torch.optim.Adam(model.parameters(), lr=opt.LR)
    scheduler         = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    dataset           = Dataset(wav_data)
    running_loss      = 0.0
    finished_process  = opt.STARTING_NUMBER

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training {num_trainable_params} parameters")

    while True:
        input_tensor, code_tensor, code = dataset.get()
        input_tensor = input_tensor.to(dml, non_blocking=True)
        optimizer.zero_grad()

        output_tensor = model(input_tensor)
        output_tensor = output_tensor.log_softmax(2).permute(1, 0, 2).to('cpu')

        code_decoded = utilities.decode_ctc(output_tensor)

        input_lengths = torch.full((output_tensor.size(1),), output_tensor.size(0), dtype=torch.long)
        target_lengths = torch.tensor([code_tensor.size(0)], dtype=torch.long)
        loss = loss_fn(output_tensor, code_tensor, input_lengths, target_lengths)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        finished_process += 1

        if finished_process % opt.PRINT_RESULTS == 0:
            print(f"Loss: {loss.item():.10f}, output: {code_decoded}, expected: {code}, output length: {len(output_tensor)}")

        if finished_process % opt.SAVE_MODEL_AFTER == 0:
            print('='*120)
            save_model(model, str(finished_process), dml)


def save_model(model_instance, name_of_save: str, dml):
    model.to('cpu', non_blocking=True)
    # Current save
    if pathlib.Path.exists(model_path_manager.root_path / (name_of_save + '.pth')):
        pathlib.Path.unlink(model_path_manager.root_path / (name_of_save + '.pth'))
    torch.save(model_instance.state_dict(), model_path_manager.root_path / (name_of_save + '.pth'))

    # Latest save
    if pathlib.Path.exists(model_path_manager.root_path / 'latest.pth'):
        pathlib.Path.unlink(model_path_manager.root_path / 'latest.pth')
    torch.save(model_instance.state_dict(), model_path_manager.root_path / 'latest.pth')
    model.to(dml, non_blocking=True)


if __name__ == "__main__":
    model_path_manager = PathManager(opt.MODEL)
    loss_file = open(model_path_manager.loss_file, 'a+')
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    start_message = '=' * 55 + f'{dt_string}' + '=' * 55
    loss_file.write(start_message + '\n')
    loss_file.close()

    while True:
        try:
            model = DTMFNet()
            torch.autograd.set_detect_anomaly(True)
            if opt.CONTINUE_LEARNING:
                print("Loading model from state dict..")
                model.load_state_dict(torch.load(model_path_manager.root_path / 'latest.pth'))

            train_model(model, nn.CTCLoss(blank=0))
        except Exception as e:
            print(f"Training crashed due to error {e}")