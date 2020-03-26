import argparse
from classifier_utils import *


def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='path to')
    parser.add_argument('checkpoint', type=str, help='checkpoint path')
    parser.add_argument('--top_k', type=int, default=5, help='top k most probable classes')
    parser.add_argument('--category_names', type=str, default=' ', help='use a mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='enable device training on gpu')
    return parser.parse_args()


def main():
    is_device_gpu = True
    args = get_input_args()
    if args.gpu:
        is_device_gpu = cuda.is_available()
    model = load_checkpoint(args.checkpoint)
    image = process_image(args.image)
    top_probs, top_classes, top_classes_index = predict(image, model, is_device_gpu, args.top_k)

    if args.category_names != ' ':
        print('index=>{}'.format(top_classes_index))
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)

        for result in range(len(top_probs)):
            print('Order={:<2}| Class={:<4}| Probability=> {:.4f}\n'.format(result + 1, top_classes[result],
                                                                            top_probs[result]))
    else:
        for result in range(len(top_probs)):
            print('Order={:<2}| Class={:<4}| Probability=> {:.4f}\n'.format(result + 1, top_classes[result],
                                                                            top_probs[result]))


if __name__ == "__main__":
    main()
