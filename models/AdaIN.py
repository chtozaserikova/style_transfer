import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_mean_std(feat, eps=1e-5):
    """
    Вычисляет среднее и стандартное отклонение для признаков.
    
    Args:
        feat (torch.Tensor): Входной тензор признаков размером (N, C, H, W)
        eps (float): Малое значение для стабилизации вычислений
    
    Returns:
        tuple: (среднее значение, стандартное отклонение)
    """
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    """Нормализует признаки по среднему и дисперсии."""
    mean, std = calc_mean_std(feat)
    return (feat - mean) / std


def _calc_feat_flatten_mean_std(feat):
    """Вычисляет среднее и стандартное отклонение для 'развернутого' тензора признаков."""
    C, H, W = feat.size()
    feat_flatten = feat.view(C, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


# Декодер для преобразования стиля
decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

# Модель VGG для извлечения признаков
vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1 (последний используемый слой)
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


class AdaIN(nn.Module):
    """Реализация Adaptive Instance Normalization (AdaIN)."""
    
    def forward(self, content, style):
        """Применяет адаптивную нормализацию к контентному изображению на основе стиля."""
        content_mean, content_std = self.calc_mean_std(content)
        style_mean, style_std = self.calc_mean_std(style)

        normalized_content = (content - content_mean) / content_std
        return normalized_content * style_std + style_mean

    @staticmethod
    def calc_mean_std(feat, eps=1e-5):
        """Вычисляет среднее и стандартное отклонение для признаков."""
        size = feat.size()
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std


class Transform(nn.Module):
    """Модуль трансформации для переноса стиля."""
    
    def __init__(self, in_planes):
        super().__init__()
        self.adain = AdaIN()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1):
        """Применяет перенос стиля и объединяет признаки разных уровней."""
        adain_output_4_1 = self.adain(content4_1, style4_1)
        adain_output_5_1 = self.adain(content5_1, style5_1)

        upsampled_5_1 = self.upsample(adain_output_5_1)
        
        # Объединение признаков
        merged = self.merge_conv(self.merge_conv_pad(adain_output_4_1 + upsampled_5_1))
        return merged


def calc_content_loss(input, target):
    """Вычисляет loss для контента (MSE между входом и целью)."""
    assert input.size() == target.size()
    return F.mse_loss(input, target)


def calc_style_loss(input, target):
    """Вычисляет loss для стиля (MSE между средними и std признаков)."""
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return F.mse_loss(input_mean, target_mean) + F.mse_loss(input_std, target_std)


def adaptive_instance_normalization(content_feat, style_feat):
    """Адаптивная нормализация экземпляров (AdaIN)."""
    assert content_feat.size()[:2] == style_feat.size()[:2]
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                  interpolation_weights=None, device='cpu'):
    """
    Выполняет перенос стиля с использованием AdaIN.
    
    Args:
        vgg (nn.Module): Модель VGG для извлечения признаков
        decoder (nn.Module): Декодер для генерации изображения
        content (torch.Tensor): Контентное изображение
        style (torch.Tensor): Стилевое изображение
        alpha (float): Коэффициент смешивания (0-1)
        interpolation_weights (list): Веса для интерполяции стилей
        device (str): Устройство для вычислений
    
    Returns:
        tuple: (результирующее изображение, content loss, style loss)
    """
    assert 0.0 <= alpha <= 1.0
    
    # Извлечение признаков
    content_f = vgg(content)
    style_f = vgg(style)
    
    # Применение AdaIN
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.zeros(1, C, H, W, device=device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat += w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    
    # Смешивание с исходным контентом
    feat = feat * alpha + content_f * (1 - alpha)
    
    # Декодирование и вычисление loss
    output = decoder(feat)
    content_loss = calc_content_loss(content, output)
    style_loss = calc_style_loss(style, output)

    return output, content_loss, style_loss