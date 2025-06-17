import matplotlib.pyplot as plt
import numpy as np
import torch

WIKIART_STYLE_MAP = {
    'Baroque': 1,
    'Cubism': 2,
    'Early_Renaissance': 3,
    'Pointillism': 4,
    'Ukiyo_e': 5,
}


from skimage.feature import canny
from skimage.metrics import structural_similarity
from skimage.filters import sobel

def compute_ssim(content_images, stylised_images):
    """Returns the average structural similarity (SSIM) between the content and stylised images"""
    ssim_sum = 0
    for content, stylised in zip(content_images, stylised_images):
        
        # Convert to numpy array
        content_img = content.detach().cpu().numpy()
        stylised_img = stylised.detach().cpu().numpy()
        
        # Convert image to grayscale
        gray_content = content_img[0, :, :]
        gray_stylised = stylised_img[0, :, :]
        
        # Compute SSIM
        score = structural_similarity(gray_content, gray_stylised, data_range=1.0)
        ssim_sum += score
    
    return ssim_sum

def plot_results(content_images, style_images, style_labels, stylised_images, nrows=5, model_name="", set_edge_descriptor=True, msgnet=False, one_of_each=True):
    """Plot the stylisation results with Sobel or Canny edge detection
    
    Args:
        content_images (tensor -> shape(B, C, H, W)): Defines the content images
        style_images (tensor -> shape(B, C, H, W)): Defines the style images
        style_labels (tensor -> shape(B)): Defines the labels of the style images
        stylised_images (tensor -> shape(B, C, H, W)): Defines the output stylised images
        nrows (int): Defines the number of samples to plot
        model_name (string): Defines the model name to print on the plot
        set_edge_descriptor (boolean): Defines the edge detection method to use. True = Sobel, False = Canny edge
        msgnet (boolean): Asserted if stylised_img is to be converted into PIL Image
        one_of_each (boolean): Asserted to plot one of each style class
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=5, figsize=(20, 20), subplot_kw={'xticks': [], 'yticks': []})
    
    if one_of_each:
        tmp_labels = []
        tmp_content = []
        tmp_styles = []
        tmp_outputs = []
        for idx, label in enumerate(style_labels):
            if label not in tmp_labels:
                tmp_labels.append(label)
                tmp_content.append(content_images[idx])
                tmp_styles.append(style_images[idx])
                tmp_outputs.append(stylised_images[idx])
        if len(tmp_labels) != nrows:
            print(style_labels)
            raise ValueError("Number of unique classes found in this batch did not match the number of classes")
        content_images = tmp_content
        style_images = tmp_styles
        style_labels = tmp_labels
        stylised_images = tmp_outputs

    for i in range(nrows):
        content_img = content_images[i].cpu()
        style_img = style_images[i].cpu()
        label = style_labels[i].cpu()
        
        # Plot content, style and stylised images
        axes[i][0].imshow(content_img.permute(1, 2, 0))
        axes[i][1].imshow(style_img.permute(1, 2, 0))
        if msgnet:
            stylised_img = stylised_images[i]
            axes[i][2].imshow(stylised_img)
        else:
            stylised_img = stylised_images[i].cpu()
            axes[i][2].imshow(stylised_img.permute(1, 2, 0))
        
        label = list(WIKIART_STYLE_MAP.keys())[list(WIKIART_STYLE_MAP.values()).index(label)]
        axes[i][1].set_title(label, size=12)
        
        # Plot canny edges of content and stylised image
        content_img = content_img.permute(1, 2, 0).numpy()[:, :, 0]
        
        if not msgnet:
            stylised_img = stylised_img.permute(1, 2, 0).numpy()[:, :, 0]
        
        if set_edge_descriptor:
            axes[i][3].imshow(sobel(content_img), cmap="copper")
            if msgnet:
                axes[i][4].imshow(sobel(np.asarray(stylised_img.convert('L'))), cmap="copper")
            else:  
                axes[i][4].imshow(sobel(stylised_img), cmap="copper")
            edge_descriptor_title = 'Sobel'
        else:
            axes[i][3].imshow(canny(content_img, sigma=1), cmap="copper")
            if msgnet:
                axes[i][4].imshow(canny(np.asarray(stylised_img.convert('L')), sigma=1), cmap="copper")
            else:
                axes[i][4].imshow(canny(stylised_img, sigma=1), cmap="copper")

            edge_descriptor_title = 'Canny'
        # Set subplot titles
        if i == nrows - 1:
            axes[i][0].text(0.5, -0.07, "Content", size=16, ha="center", transform=axes[i][0].transAxes)
            axes[i][1].text(0.5, -0.07, "Style", size=16, ha="center", transform=axes[i][1].transAxes)
            axes[i][2].text(0.5, -0.07, f"{model_name}", size=16, ha="center", transform=axes[i][2].transAxes)
            
            axes[i][3].text(0.5, -0.07, f"Content {edge_descriptor_title} Edges", size=16, ha="center", transform=axes[i][3].transAxes)
            axes[i][4].text(0.5, -0.07, f"{model_name} {edge_descriptor_title} Edges", size=16, ha="center", transform=axes[i][4].transAxes)

    fig.suptitle(f'{model_name} Results', size=21, y=0.99)
    plt.tight_layout()
    plt.show()
    
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import numpy as np
from models.Inception import InceptionV3

def get_activations(
    data, model, dims=2048, device="cpu", num_workers=1
):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    pred_arr = np.empty((len(data), dims))

    start_idx = 0

    for batch in data:
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.unsqueeze(0))[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx : start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError("Imaginary component {}".format(m))
            return 0
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(
    data, model, dims=2048, device="cpu", num_workers=3
):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(data, model, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_fid_from_dataset(inputs, outputs, device, dims=2048, num_workers=3):
    """calculates fid from images in our dataset"""
    block_idx = InceptionV3().BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx], resize_input=False).to(device)

    m1, s1 = calculate_activation_statistics(
        inputs, model, dims, device, num_workers
    )
    m2, s2 = calculate_activation_statistics(
        outputs, model, dims, device, num_workers
    )
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def calc_content_loss(input, target):
    assert (input.size() == target.size())
    return torch.nn.MSELoss()(input, target)

def calc_style_loss(input, target):
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return torch.nn.MSELoss()(input_mean, target_mean) + torch.nn.MSELoss()(input_std, target_std)

def plot_training_history(content_losses, style_losses, total_losses):
    '''
    Plot training history with two plots
        1) Content loss vs. Style loss
        2) Total loss
    '''
    plt.figure()
    plt.plot(content_losses)
    plt.plot(style_losses)
    plt.title('Content loss vs Style loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Content', 'Style'], loc='upper left')

    plt.figure()
    plt.plot(total_losses)
    plt.title('Total loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.show()
    
def plot_cyclegan_training(D_losses, G_losses, adv_losses, cycle_losses, I_losses):
    '''
    Plot training history of CycleGAN
    '''
    plt.figure()
    plt.plot(D_losses)
    plt.plot(G_losses)
    plt.plot(adv_losses)
    plt.plot(cycle_losses)
    plt.plot(I_losses)
    plt.title('Losses')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Discriminator', 'Generator', 'GAN', 'Cycle', 'Identity'], loc='upper left')

    plt.figure()
    plt.plot(D_losses)
    plt.plot(adv_losses)
    plt.plot(cycle_losses)
    plt.plot(I_losses)
    plt.title('Losses')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Discriminator', 'GAN', 'Cycle', 'Identity'], loc='upper left')

    plt.show()