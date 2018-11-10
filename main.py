from RoverAPI import *
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name',
    	type=str,
    	help='path to the model file if autonomous.')

    parser.add_argument(
    	'--autonomous',
    	type=bool,
    	default=False,
    	help='True for autonomous, False for human control. Default False.')

    parser.add_argument(
       '--network',
       type=str,
       help='Name of the network you want to run if autonomous. Ex. resnet34')

    parser.add_argument(
        '--driver',
        type=str,
        default='unknown_driver',
        help='The name of the person operating or running the rover. Optional')

    parser.add_argument(
        '--rover',
        type=str,
        default='no_name',
        help='The name on the rover being used. Optional')

    parser.add_argument(
        '--frames_per_second',
        type=int,
        default=30,
        help='The frame rate the rover will be operating at. Default 30')

    parser.add_argument(
        '--show_video_feed',
        type=bool,
        default=False,
        help="True to see rover's video feed, False to supress this feature.")

    parser.add_argument(
        '--save_training_data',
        type=bool,
        default=False,
        help='y to save training data if not autonomous, n to not save. Default y')

    parser.add_argument(
        '--ml_framework',
        type=str,
        default='tf',
        help='tf for TensorFlow model, pt for PyTorch model. Default tf')

    parser.add_argument(
        '--image_type',
        type=str,
        default='color',
        help='grayscale, color, or framestack for model input if autonomous. Default is color.')

    parser.add_argument(
        '--norm_method',
        type=str,
        default=None,
        help='Type instance_norm or channel_norm. Default None')

    parser.add_argument(
        '--norm_vals',
        type=str,
        default='0,0,0',
        help='values to use in normalization if norm_method is not None.')

    parser.add_argument(
        '--num_outputs',
        type=int,
        default=4,
        help='The number of outputs for the network. Default 4.')

    args = parser.parse_args()
    norm_vals = [int(item) for item in args.norm_vals.split(',')]

    if args.save_training_data is True and args.autonomous is True:
        args.save_training_data = False

    rover = RoverRun(args.model_name,
                    args.network,
                    args.autonomous,
                    args.driver,
                    args.rover,
                    args.frame_per_second,
                    args.show_video_feed,
                    args.save_training_data,
                    args.ml_framework,
                    args.image_type,
                    args.norm_method,
                    norm_vals,
                    args.num_outputs)
