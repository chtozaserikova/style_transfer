import torch.nn as nn
import torch
from utils.eval import calc_content_loss, calc_style_loss


def calc_mean_std(feat, eps=1e-5):
    """Вычисляет среднее и стандартное отклонение для feature map.
    
    Args:
        feat (torch.Tensor): Входной тензор размерности (N, C, H, W)
        eps (float): Малое значение для избежания деления на ноль
        
    Returns:
        Кортеж (mean, std) - среднее и стандартное отклонение
    """
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    """Нормализует feature map по среднему и дисперсии."""
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


def _calc_feat_flatten_mean_std(feat):
    """Вычисляет среднее и стандартное отклонение для 3D feature map (C, H, W)."""
    assert feat.size()[0] == 3
    assert isinstance(feat, torch.FloatTensor)
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


# Декодер - преобразует features обратно в изображение
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

# VGG-19 encoder (до relu5_1)
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
    nn.ReLU(),  # relu4-1
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


class SANet(nn.Module):
    """Модуль Self-Attention Network для переноса стиля."""
    
    def __init__(self, in_planes):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))  # преобразование для контента
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))  # преобразование для стиля
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))  # преобразование перед выходом
        self.sm = nn.Softmax(dim=-1)  # softmax для матрицы внимания
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))  # финальное преобразование

    def forward(self, content, style):
        # Нормализуем feature maps
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)
        
        # Преобразуем feature maps для матричного умножения
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        b, c, h, w = G.size()
        G = G.view(b, -1, w * h)
        
        # Вычисляем матрицу внимания
        S = torch.bmm(F, G)
        S = self.sm(S)
        
        # Применяем attention к стилю
        b, c, h, w = H.size()
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))
        
        # Восстанавливаем исходную размерность и добавляем residual connection
        b, c, h, w = content.size()
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        O += content
        
        return O


class Transform(nn.Module):
    """Основной модуль трансформации стиля."""
    
    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.sanet4_1 = SANet(in_planes=in_planes)  # для слоя relu4_1
        self.sanet5_1 = SANet(in_planes=in_planes)  # для слоя relu5_1
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')  # апсемплинг
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))  # объединяющая свертка

    def forward(self, content4_1, style4_1, content5_1, style5_1):
        # Применяем SANet к двум уровням features и объединяем результаты
        return self.merge_conv(
            self.merge_conv_pad(
                self.sanet4_1(content4_1, style4_1) + 
                self.upsample5_1(self.sanet5_1(content5_1, style5_1))
            )
        )


class Net(nn.Module):
    """Основная сеть для переноса стиля."""
    
    def __init__(self, encoder, decoder, start_iter):
        super(Net, self).__init__()
        # Разделяем encoder на части по слоям
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        
        self.transform = Transform(in_planes=512)  # модуль трансформации
        self.decoder = decoder  # декодер
        
        # Загрузка весов если продолжаем обучение
        if start_iter > 0:
            self.transform.load_state_dict(
                torch.load('transformer_iter_' + str(start_iter) + '.pth'))
            self.decoder.load_state_dict(
                torch.load('decoder_iter_' + str(start_iter) + '.pth'))
        
        self.mse_loss = nn.MSELoss()
        
        # Фиксируем веса encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def encode_with_intermediate(self, input):
        """Кодирует изображение с возвратом промежуточных feature maps."""
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target, norm=False):
        """Вычисляет loss для контента."""
        if not norm:
            return self.mse_loss(input, target)
        else:
            return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))

    def calc_style_loss(self, input, target):
        """Вычисляет loss для стиля."""
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    
    def forward(self, content, style):
        """Основной forward pass."""
        # Получаем feature maps для контента и стиля
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        
        # Применяем трансформацию стиля
        stylized = self.transform(
            content_feats[3], style_feats[3], 
            content_feats[4], style_feats[4]
        )
        
        # Декодируем результат
        g_t = self.decoder(stylized)
        g_t_feats = self.encode_with_intermediate(g_t)
        
        # Вычисляем loss для контента
        loss_c = self.calc_content_loss(g_t_feats[3], content_feats[3], norm=True) + \
                 self.calc_content_loss(g_t_feats[4], content_feats[4], norm=True)
        
        # Вычисляем loss для стиля
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        
        # Identity losses
        Icc = self.decoder(self.transform(
            content_feats[3], content_feats[3], 
            content_feats[4], content_feats[4]
        ))
        Iss = self.decoder(self.transform(
            style_feats[3], style_feats[3], 
            style_feats[4], style_feats[4]
        ))
        
        l_identity1 = self.calc_content_loss(Icc, content) + \
                      self.calc_content_loss(Iss, style)
        
        Fcc = self.encode_with_intermediate(Icc)
        Fss = self.encode_with_intermediate(Iss)
        
        l_identity2 = self.calc_content_loss(Fcc[0], content_feats[0]) + \
                      self.calc_content_loss(Fss[0], style_feats[0])
        
        for i in range(1, 5):
            l_identity2 += self.calc_content_loss(Fcc[i], content_feats[i]) + \
                           self.calc_content_loss(Fss[i], style_feats[i])
        
        return loss_c, loss_s, l_identity1, l_identity2


def compute_style_loss(style_images, stylised_images, net):
    """Вычисляет стилевой loss между стилевыми и стилизованными изображениями.
    
    Args:
        style_images: Исходные стилевые изображения
        stylised_images: Стилизованные изображения
        net: Модель encoder (VGG)
        
    Returns:
        Средний стилевой loss по всем слоям
    """
    enc_1, enc_2, enc_3, enc_4, enc_5 = net[0], net[1], net[2], net[3], net[4]
    loss_s = 0.0
    
    # Вычисляем loss для каждого слоя
    output1_1 = enc_1(style_images)
    style1_1 = enc_1(stylised_images) 
    loss_s += calc_style_loss(output1_1, style1_1)
    
    output2_1 = enc_2(output1_1)
    style2_1 = enc_2(style1_1)
    loss_s += calc_style_loss(output2_1, style2_1)

    output3_1 = enc_3(output2_1)
    style3_1 = enc_3(style2_1)
    loss_s += calc_style_loss(output3_1, style3_1)

    output4_1 = enc_4(output3_1)
    style4_1 = enc_4(style3_1)
    loss_s += calc_style_loss(output4_1, style4_1)

    output5_1 = enc_5(output4_1)
    style5_1 = enc_5(style4_1)
    loss_s += calc_style_loss(output5_1, style5_1)
    
    return float(loss_s / 5)  # Возвращаем средний loss


def compute_content_loss(content_images, stylised_images, net):
    """Вычисляет контентный loss между контентными и стилизованными изображениями.
    
    Args:
        content_images: Исходные контентные изображения
        stylised_images: Стилизованные изображения
        net: Модель encoder (VGG)
        
    Returns:
        Средний контентный loss по всем слоям
    """
    enc_1, enc_2, enc_3, enc_4, enc_5 = net[0], net[1], net[2], net[3], net[4]
    loss_c = 0.0

    # Вычисляем loss для глубоких слоев
    output1 = enc_4(enc_3(enc_2(enc_1(content_images))))
    content1 = enc_4(enc_3(enc_2(enc_1(stylised_images))))
    loss_c += calc_content_loss(output1, content1)
    
    output2 = enc_5(output1)
    content2 = enc_5(content1)
    loss_c += calc_content_loss(output2, content2)
        
    return float(loss_c / 2)  # Возвращаем средний loss