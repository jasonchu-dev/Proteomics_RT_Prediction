import torch

def normalize_neg_one_2_one(y):
    upper_bound = 10000
    normalized = (y / upper_bound) - 1
    return normalized

def normalize_zero_2_one(y):
    upper_bound = 20000
    normalized = y / upper_bound
    return normalized

def save_model(file_path, model, optimizer, train_loss, scaled_loss, epoch):
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'train_loss': train_loss,
        'scaled_loss': scaled_loss
    }
    torch.save(checkpoint, file_path)

def unnormalize_neg_one_2_one(prediction):
    upper_bound = 10000
    unnormalized = (prediction + 1) * upper_bound
    return unnormalized

def unnormalize_zero_2_one(prediction):
    upper_bound = 20000
    unnormalized = prediction * upper_bound
    return unnormalized

def format_number(num, length=20):
    num_str = str(num)
    if "." in num_str:
        integer_part, decimal_part = num_str.split(".")
        formatted_decimal_part = decimal_part.ljust(3, '0')
        formatted_integer_part = integer_part
        formatted_num = "{}.{}".format(formatted_integer_part, formatted_decimal_part)
    else:
        formatted_num = num_str
    formatted_num = formatted_num.ljust(length, '0')
    return formatted_num