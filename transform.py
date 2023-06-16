from networks import *

if __name__=='__main__':
    if(len(sys.argv) < 2):
        print("Usage: make transform IMAGE=PATH_TO_IMAGE_FILENAME")
        exit(0)
    if not (os.path.isfile(sys.argv[1])):
        print("{} is not a file".format(sys.argv[1]))
        exit(0)
    if not (os.path.isfile('generator_release.pth')):
        print('Can not find pre-trained weights file generator_release.pth. Please provide within current directory.')
        exit(0)
    checkpoint = torch.load('checkpoint_epoch_101.pth', map_location='cpu')
    G = Generator().to('cpu')
    G.load_state_dict(checkpoint['g_state_dict'])

    checkpoint2 = torch.load('generator_release.pth', map_location='cpu')
    G2 = Generator().to('cpu')
    G2.load_state_dict(checkpoint2['g_state_dict'])

    transformer = transforms.Compose([
        transforms.ToTensor()
        ])

    with Image.open(sys.argv[1]) as img:
        # The input is needed as a batch, I got the solution from here:
        # https://discuss.pytorch.org/t/pytorch-1-0-how-to-predict-single-images-mnist-example/32394
        pseudo_batched_img = transformer(img)
        pseudo_batched_img = pseudo_batched_img[None]
        result = G(pseudo_batched_img)
        result = transforms.ToPILImage()(result[0]).convert('RGB')
        result.save('transformed_2block.jpg')

        result2 = G2(pseudo_batched_img)
        result2 = transforms.ToPILImage()(result2[0]).convert('RGB')
        result2.save('transformed_original.jpg')

