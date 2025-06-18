import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# Веса модели Inception, перенесенные из TensorFlow в PyTorch
# Оригинальный источник: http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
FID_WEIGHTS_URL = "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth"  # noqa: E501


class InceptionV3(nn.Module):
    """Предобученная сеть InceptionV3, возвращающая карты признаков"""

    # Индекс блока по умолчанию для возврата (соответствует выходу последнего avg pooling)
    DEFAULT_BLOCK_INDEX = 3

    # Соответствие размерности признаков индексам выходных блоков
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # Признаки после первого max pooling
        192: 1,  # Признаки после второго max pooling
        768: 2,  # Признаки перед вспомогательным классификатором
        2048: 3, # Признаки после финального avg pooling
    }

    def __init__(
        self,
        output_blocks=(DEFAULT_BLOCK_INDEX,),
        resize_input=True,
        normalize_input=True,
        requires_grad=False,
        use_fid_inception=True,
    ):
        """Инициализация предобученной модели InceptionV3

        Параметры
        ----------
        output_blocks : list of int
            Индексы блоков, признаки которых нужно возвращать. Возможные значения:
                - 0: выход первого max pooling
                - 1: выход второго max pooling
                - 2: выход перед вспомогательным классификатором
                - 3: выход финального avg pooling
        resize_input : bool
            Если True, входное изображение билинейно масштабируется до 299x299 пикселей
            перед подачей в модель. Так как сеть без полносвязных слоев полностью
            свёрточная, она может обрабатывать изображения любого размера,
            поэтому масштабирование может быть не обязательным
        normalize_input : bool
            Если True, масштабирует вход из диапазона (0, 1) в диапазон, ожидаемый
            предобученной сетью Inception (-1, 1)
        requires_grad : bool
            Если True, параметры модели требуют вычисления градиентов. Полезно для
            дообучения сети
        use_fid_inception : bool
            Если True, использует предобученную модель Inception из реализации FID
            в TensorFlow. Если False, использует стандартную предобученную модель
            из torchvision. Модель для FID имеет другие веса и немного другую
            архитектуру. Для вычисления FID-метрики рекомендуется установить True,
            чтобы получить сравнимые результаты.
        """
        super().__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, "Максимальный возможный индекс выходного блока - 3"

        self.blocks = nn.ModuleList()

        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = _inception_v3(weights="DEFAULT")

        # Блок 0: от входа до первого maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Блок 1: от maxpool1 до maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2),
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Блок 2: от maxpool2 до вспомогательного классификатора
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Блок 3: от вспомогательного классификатора до финального avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Получить карты признаков Inception

        Параметры
        ----------
        inp : torch.Tensor
            Входной тензор размерности Bx3xHxW. Ожидаются значения в диапазоне (0, 1)

        Возвращает
        -------
        List[torch.Tensor]
            Список тензоров с картами признаков выбранных блоков,
            отсортированный по возрастанию индекса
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Масштабирование из (0, 1) в (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


def _inception_v3(*args, **kwargs):
    """Обёртка для `torchvision.models.inception_v3`"""
    try:
        version = tuple(map(int, torchvision.__version__.split(".")[:2]))
    except ValueError:
        # Защита от нестандартных строк версии
        version = (0,)

    # Пропуск инициализации весов по умолчанию, если поддерживается версией torchvision
    # См. https://github.com/mseitzer/pytorch-fid/issues/28.
    if version >= (0, 6):
        kwargs["init_weights"] = False

    # Обратная совместимость: аргумент `weights` обрабатывался через `pretrained`
    # в версиях до 0.13.
    if version < (0, 13) and "weights" in kwargs:
        if kwargs["weights"] == "DEFAULT":
            kwargs["pretrained"] = True
        elif kwargs["weights"] is None:
            kwargs["pretrained"] = False
        else:
            raise ValueError(
                f"weights=={kwargs['weights']} не поддерживается в torchvision {torchvision.__version__}"
            )
        del kwargs["weights"]

    return torchvision.models.inception_v3(*args, **kwargs)


def fid_inception_v3():
    """Создание предобученной модели Inception для вычисления FID

    Модель Inception для FID использует другие веса и немного отличается
    от стандартной Inception из torchvision.

    Этот метод сначала создаёт стандартную Inception, а затем заменяет
    необходимые части, которые отличаются в модели для FID.
    """
    inception = _inception_v3(num_classes=1008, aux_logits=False, weights=None)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)

    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception


class FIDInceptionA(torchvision.models.inception.InceptionA):
    """Блок InceptionA с изменениями для вычисления FID"""

    def __init__(self, in_channels, pool_features):
        super().__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Изменение: avg_pool2d в TensorFlow не учитывает нули добавленные padding'ом
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(torchvision.models.inception.InceptionC):
    """Блок InceptionC с изменениями для вычисления FID"""

    def __init__(self, in_channels, channels_7x7):
        super().__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # Изменение: avg_pool2d в TensorFlow не учитывает нули добавленные padding'ом
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(torchvision.models.inception.InceptionE):
    """Первый блок InceptionE с изменениями для вычисления FID"""

    def __init__(self, in_channels):
        super().__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Изменение: avg_pool2d в TensorFlow не учитывает нули добавленные padding'ом
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(torchvision.models.inception.InceptionE):
    """Второй блок InceptionE с изменениями для вычисления FID"""

    def __init__(self, in_channels):
        super().__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Изменение: В модели FID используется max pooling вместо average pooling.
        # Вероятно, это ошибка в данной реализации Inception, так как в других
        # реализациях Inception используется average pooling (как и описано в статье).
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)