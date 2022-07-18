import random

from PIL import Image


# class Compose(object):
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, img, mask):
#         assert img.size == mask.size
#         for t in self.transforms:
#             img, mask = t(img, mask)
#         return img, mask


# class RandomHorizontallyFlip(object):
#     def __call__(self, img, mask):
#         if random.random() < 0.5:
#             return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
                  
#         return img, mask


# class Resize(object):
#     def __init__(self, size):
#         self.size = tuple(reversed(size))  # size: (h, w)

#     def __call__(self, img, mask):
#         assert img.size == mask.size
#         return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)
             

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, dst1, dst2):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask, dst1, dst2 = t(img, mask, dst1, dst2)
        return img, mask, dst1, dst2


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, dst1, dst2):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), \
                   dst1.transpose(Image.FLIP_LEFT_RIGHT), dst2.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask, dst1, dst2

class Resize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask, dst1, dst2):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST), \
               dst1.resize(self.size, Image.BILINEAR), dst2.resize(self.size, Image.NEAREST)


# class Compose(object):
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, img, mask, dst1, dst2,img2,img3):
#         assert img.size == mask.size
#         for t in self.transforms:
#             img, mask, dst1, dst2,img2,img3 = t(img, mask, dst1, dst2,img2,img3)
#         return img, mask, dst1, dst2, img2,img3


# class RandomHorizontallyFlip(object):
#     def __call__(self, img, mask, dst1, dst2, img2,img3):
#         if random.random() < 0.5:
#             return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), \
#                    dst1.transpose(Image.FLIP_LEFT_RIGHT), dst2.transpose(Image.FLIP_LEFT_RIGHT), img2.transpose(Image.FLIP_LEFT_RIGHT), img3.transpose(Image.FLIP_LEFT_RIGHT)
#         return img, mask, dst1, dst2, img2,img3


# class Resize(object):
#     def __init__(self, size):
#         self.size = tuple(reversed(size))  # size: (h, w)

#     def __call__(self, img, mask, dst1, dst2, img2,img3):
#         assert img.size == mask.size
#         return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST), \
#                dst1.resize(self.size, Image.NEAREST), dst2.resize(self.size, Image.NEAREST),  img2.resize(self.size, Image.NEAREST), img3.resize(self.size, Image.NEAREST)