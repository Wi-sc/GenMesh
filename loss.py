import torch
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    # window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def get_ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def get_iou(outputs: torch.Tensor, labels: torch.Tensor, size_average=True):

    bacth_size = labels.shape[0]

    inter = (outputs * labels).view(bacth_size, -1).sum(1) # B

    union = (outputs + labels).view(bacth_size, -1).sum(1) - inter # B

    iou = inter / (union + 1e-6)
    
    if size_average:
        return iou.mean()
    else:
        return iou

def get_iou_loss(outputs: torch.Tensor, labels: torch.Tensor):

    bacth_size = labels.shape[0]

    inter = (outputs * labels).view(bacth_size, -1).sum(1) # B

    union = (outputs + labels).view(bacth_size, -1).sum(1) - inter # B

    iou = inter / (union + 1e-6)

    return 1-iou.mean()


def batch_contrastive_loss(hidden1: torch.Tensor,
                    hidden2: torch.Tensor,
                    hidden_norm: bool = True,
                    temperature: float = 1.0):
    """
    hidden1: (batch_size, dim)
    hidden2: (batch_size, dim)
    """
    batch_size, hidden_dim = hidden1.shape
    
    if hidden_norm:
        hidden1 = torch.nn.functional.normalize(hidden1, p=2, dim=-1)
        hidden2 = torch.nn.functional.normalize(hidden2, p=2, dim=-1)

    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = torch.arange(0, batch_size).to(device=hidden1.device)
    masks = torch.nn.functional.one_hot(torch.arange(0, batch_size), num_classes=batch_size).to(device=hidden1.device, dtype=torch.float)

    # logits_aa = torch.matmul(hidden1, hidden1_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)
    # logits_aa = logits_aa - masks * 1e9
    # logits_bb = torch.matmul(hidden2, hidden2_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)
    # logits_bb = logits_bb - masks * 1e9
    logits_ab = torch.matmul(hidden1, hidden2_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)
    logits_ba = torch.matmul(hidden2, hidden1_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)

    # loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
    # loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
    loss_a = torch.nn.functional.cross_entropy(logits_ab, labels)
    loss_b = torch.nn.functional.cross_entropy(logits_ba, labels)
    loss = loss_a + loss_b
    return loss

def instance_contrastive_loss(hidden1: torch.Tensor,
                    hidden2: torch.Tensor,
                    hidden_norm: bool = True,
                    temperature: float = 1.0):
    """
    hidden1: (batch_size, points, dim)
    hidden2: (batch_size, points, dim)
    """
    batch_size, points_num, hidden_dim = hidden1.shape
    
    if hidden_norm:
        hidden1 = torch.nn.functional.normalize(hidden1, p=2, dim=-1)
        hidden2 = torch.nn.functional.normalize(hidden2, p=2, dim=-1)

    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = torch.arange(0, points_num).to(device=hidden1.device).unsqueeze(0).expand(batch_size, -1).contiguous().view(-1)
    masks = torch.nn.functional.one_hot(torch.arange(0, points_num), num_classes=points_num).to(device=hidden1.device, dtype=torch.float).unsqueeze(0).expand(batch_size, -1, -1)


    logits_aa = torch.matmul(hidden1, hidden1_large.transpose(1, 2)) / temperature  # shape (batch_size, points_num, points_num)
    logits_aa = logits_aa - masks * 1e9
    logits_bb = torch.matmul(hidden2, hidden2_large.transpose(1, 2)) / temperature  # shape (batch_size, points_num, points_num)
    logits_bb = logits_bb - masks * 1e9
    logits_ab = torch.matmul(hidden1, hidden2_large.transpose(1, 2)) / temperature  # shape (batch_size, points_num, points_num)
    logits_ba = torch.matmul(hidden2, hidden1_large.transpose(1, 2)) / temperature  # shape (batch_size, points_num, points_num)
    
    loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], dim=-1).view(-1, points_num*2), labels)
    loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], dim=-1).view(-1, points_num*2), labels)
    loss = loss_a + loss_b
    return loss

def get_element_iou_loss(masks):
    batch_size, view_num, h, w = masks.shape
    elements_num = 8
    batch_size = batch_size//elements_num
    
    masks = masks.reshape(batch_size, elements_num, view_num, h, w)

    index_range = torch.arange(0, elements_num)
    a_index, b_index = torch.meshgrid(index_range, index_range)
    triu_index = torch.triu(torch.ones(elements_num, elements_num), diagonal=1)
    a_index = a_index[triu_index>0]
    b_index = b_index[triu_index>0]

    a_masks = masks[:, a_index, :, :]
    b_masks = masks[:, b_index, :, :]
    loss = get_iou(a_masks, b_masks)

    return loss
