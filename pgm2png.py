from PIL import Image
import glob
import os
import link


def batch_images(in_dir, out_dir):
    if not os.path.exists(out_dir):
        print(out_dir, 'is not existed.')
        os.mkdir(out_dir)

    if not os.path.exists(in_dir):
        print(in_dir, 'is not existed.')
        return -1
    count = 0
    for files in glob.glob(in_dir + '/*'):
        filepath, filename = os.path.split(files)

        out_file = filename.split('.')[0] + '.png'
        im = Image.open(files)
        new_path = os.path.join(out_dir, out_file)
        print(count, ',', new_path)
        count = count + 1
        im.save(os.path.join(out_dir, out_file))

if __name__ == '__main__':
    # batch_images(link.in_dir, link.out_dir)
    batch_images("C:/Users/User/Downloads/Pedestrians/TestData_part4/Data/TestData", "C:/Users/User/Downloads/Pedestrians/Test/image_png")