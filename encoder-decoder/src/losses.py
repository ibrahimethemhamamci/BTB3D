import torch
import torch.nn.functional as F
from einops import rearrange
from torchvision import transforms
import torch.distributed as dist
from torch.distributed import nn as dist_nn
from functools import cache
import torchvision.transforms.functional as vF
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint

@cache
def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def calculate_entropy_loss(
    affinity,
    loss_type: str="softmax",
    inv_temperature: float=100.,
    sample_minimization_weight: float=1.0,
    batch_maximization_weight: float=1.0,
    use_distributed_batch_entropy: bool = True,
):
    """Calculate the entropy loss."""
    if loss_type == "softmax":
        flat_affinity = affinity.view(-1, affinity.shape[-1]) * inv_temperature
        target_probs = flat_affinity.softmax(dim=-1)
        log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    def calculcate_entropy_items(target_probs, log_probs, sampling_frac=1.0):
        
        # sampling frac = 0.8
        if sampling_frac < 1.0:
            num_tokens = target_probs.size(0)
            num_sampled_tokens = int(sampling_frac * num_tokens)
            rand_mask = torch.randn(num_tokens).argsort(dim = -1) < num_sampled_tokens
            target_probs = target_probs[rand_mask]
            log_probs = log_probs[rand_mask]
        else:
            target_probs = target_probs

        avg_probs = torch.mean(target_probs, dim=0)
        # print(f">> DEBUG|{target_probs.shape}|{avg_probs.shape}")
        if use_distributed_batch_entropy and is_distributed():
            avg_probs = dist_nn.all_reduce(avg_probs)
            avg_probs /= dist.get_world_size()
        else:
            avg_probs = avg_probs
        avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-5))

        sample_entropy = -torch.mean(torch.sum(target_probs * log_probs, dim=-1))
        return sample_entropy, avg_entropy
    
    sample_entropy, avg_entropy = calculcate_entropy_items(target_probs, log_probs)

    loss = (sample_minimization_weight * sample_entropy) - (
        batch_maximization_weight * avg_entropy
    )
    return loss, sample_entropy, avg_entropy

def lecam_reg(real_pred, fake_pred, ema_real_pred, ema_fake_pred):
    """Lecam loss for data-efficient and stable GAN training.

    Described in https://arxiv.org/abs/2104.03310

    Args:
      real_pred: Prediction (scalar) for the real samples.
      fake_pred: Prediction for the fake samples.
      ema_real_pred: EMA prediction (scalar)  for the real samples.
      ema_fake_pred: EMA prediction for the fake samples.

    Returns:
      Lecam regularization loss (scalar).
    """
    assert real_pred.ndim == 0
    lecam_loss = torch.mean(torch.pow(F.relu(real_pred - ema_fake_pred), 2))
    lecam_loss += torch.mean(torch.pow(F.relu(ema_real_pred - fake_pred), 2))
    return lecam_loss

def pick_video_frame(frames, frame_indices):
    # picked_num_frames = frame_indices.shape[1]
    # batch, device = video.shape[0], video.device
    # video = rearrange(video, 'b c f ... -> b f c ...')
    # batch_indices = torch.arange(batch, device = device)
    # batch_indices = rearrange(batch_indices, 'b -> b 1')
    # images = video[batch_indices, frame_indices]
    # images = rearrange(images, 'b c ... -> (b c) ...')
    images = frames[frame_indices]
    return images

def resnet50_imaagenet1k_v2_transform(img):
    # mean = (0.485, 0.456, 0.406)
    # std = (0.229, 0.224, 0.225)
    interpolation = transforms.InterpolationMode.BILINEAR
    resize_size = 232
    crop_size = 224
    antialias = True
    # img = vF.resize(img, resize_size, interpolation=interpolation, antialias=antialias)
    # img = vF.center_crop(img, crop_size)
    img = vF.resize(img, crop_size, interpolation=interpolation, antialias=antialias)
    # img = vF.normalize(img, mean=mean, std=std)
    return img

def calculate_perceptual_loss(real_perceptual_inputs, fake_perceptual_inputs, model):
    model.eval()
    # num_frames = real_perceptual_inputs.shape[0]
    # picked_indices = torch.randint(0, num_frames, [4])
    
    # real_perceptual_inputs = pick_video_frame(real_perceptual_inputs, picked_indices)
    # fake_perceptual_inputs = pick_video_frame(fake_perceptual_inputs, picked_indices)

    real_perceptual_inputs = resnet50_imaagenet1k_v2_transform(real_perceptual_inputs)
    fake_perceptual_inputs = resnet50_imaagenet1k_v2_transform(fake_perceptual_inputs)
    real_perceptual_logits = model(real_perceptual_inputs)
    fake_perceptual_logits = model(fake_perceptual_inputs)
    return F.mse_loss(real_perceptual_logits, fake_perceptual_logits)

def r1_gradient_penalty(inputs, output, penalty_cost=10.0):
    def apply(gradients, penalty_cost: float):
        # 2. Reshape and Convert to Float32
        gradients = gradients.view(gradients.shape[0], -1) # Flatten gradients
        # gradients = gradients.float() 
        # 3. Calculate Penalty
        # penalty = torch.mean(gradients.norm(2, dim=-1)**2) * penalty_cost
        penalty = torch.mean(torch.sqrt(1e-6 + torch.sum(gradients **2, dim=1))**2) * penalty_cost
        # penalty = torch.mean(torch.sum(torch.square(gradients), dim=-1)) * penalty_cost  # Square and sum along the last dimension
        return penalty
    """
    Currently we followed the magvit-v1's implementation
    https://github.com/google-research/magvit/blob/05e8cfd6559c47955793d70602d62a2f9b0bdef5/videogvt/train_lib/losses.py#L207
    TODO: find out which implementation is better

    original: https://arxiv.org/pdf/1704.00028v3
    """
    # Calculate gradients w.r.t. the network's parameters
    gradients = torch.autograd.grad(
        outputs=output, 
        inputs=inputs,  # Typically the discriminator's parameters
        grad_outputs=output.new_ones(output.shape),  # Dummy gradients for backpropagation
        create_graph=True,  # Needed for calculating higher-order gradients
        retain_graph=True,  # Keep the graph for further computations
        only_inputs=True
    )[0]  # Get the first (and only) gradient

    gradient_penalty = apply(gradients, penalty_cost)
    del gradients, output

    return gradient_penalty



def sigmoid_cross_entropy_with_logits(*, labels: torch.Tensor,
                                      logits: torch.Tensor) -> torch.Tensor:
    """Sigmoid cross entropy loss.

    We use a stable formulation that is equivalent to the one used in TensorFlow.
    The following derivation shows how we arrive at the formulation:

    .. math::
          z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
        = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
        = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
        = (1 - z) * x + log(1 + exp(-x))
        = x - x * z + log(1 + exp(-x))

    For x < 0, the following formula is more stable:
    .. math::
          x - x * z + log(1 + exp(-x))
        = log(exp(x)) - x * z + log(1 + exp(-x))
        = - x * z + log(1 + exp(x))

    We combine the two cases (x<0, x>=0) into one formula as follows:
    .. math::
        max(x, 0) - x * z + log(1 + exp(-abs(x)))

    This function is vmapped, so it is written for a single example, but can
    handle a batch of examples.

    Args:
      labels: The correct labels.
      logits: The output logits.

    Returns:
      The binary cross entropy loss for each given logit.
    """
    # The final formulation is: max(x, 0) - x * z + log(1 + exp(-abs(x)))
    # To allow computing gradients at zero, we define custom versions of max and
    # abs functions just like in tf.nn.sigmoid_cross_entropy_with_logits.
    zeros = torch.zeros_like(logits, dtype=logits.dtype)
    condition = (logits >= zeros)
    relu_logits = torch.where(condition, logits, zeros)
    neg_abs_logits = torch.where(condition, -logits, logits)
    return relu_logits - logits * labels + torch.log1p(torch.exp(neg_abs_logits))

def discriminator_loss(real_logit, fake_logit, loss_type: str):
    
    if loss_type == "hinge":
        real_loss = F.relu(1.0 - real_logit)
        fake_loss = F.relu(1.0 + fake_logit)
    elif loss_type == "non-saturating":
        if real_logit is not None:
            real_loss = sigmoid_cross_entropy_with_logits(
                labels=torch.ones_like(real_logit), logits=real_logit
            )
        else:
            real_loss = 0.0
        if fake_logit is not None:
            fake_loss = sigmoid_cross_entropy_with_logits(
                labels=torch.zeros_like(fake_logit), logits=fake_logit
            )
        else:
            fake_loss = 0.0
    else:
        raise ValueError("Generator loss {} not supported".format(loss_type))
    disc_loss = torch.mean(real_loss) + torch.mean(fake_loss)
    return disc_loss

def generator_loss(fake_logit, loss_type: str="hinge"):
    """Adds generator loss."""
    if loss_type == "hinge":
        loss = -torch.mean(fake_logit)
    elif loss_type == "non-saturating":
        loss = torch.mean(
          sigmoid_cross_entropy_with_logits(
              labels=torch.ones_like(fake_logit), logits=fake_logit))
    else:
        raise ValueError("Generator loss {} not supported".format(loss_type))
    return loss
