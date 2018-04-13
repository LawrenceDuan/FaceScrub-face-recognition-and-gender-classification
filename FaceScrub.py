import argparse
import imageDLandCROP
import seprateDataset
from scipy.misc import imread
actor = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

def get_parser():
    # Get parser for command line arguments.
    parser = argparse.ArgumentParser(description="Face Scrub")
    parser.add_argument("-n",
                        "--number",
                        dest="number")
    return parser

# Data preparation
def part0():
    imageDLandCROP.DLandCROP(actor)

# Split Dataset
def part2():
    im_data = seprateDataset.read(actor, 'cropped/')
    im_data_training, im_data_validation, im_data_testing = seprateDataset.split(actor, im_data, 70, 10, 10)
    print (len(im_data_validation[0]))

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    switcher = {
        0: part0,
        #1: part1,
        2: part2,
        #3: part3,
        #4: part4,
        #5: part5,
        #6: part6,
        #7: part7,
        #8: part8,
    }
    # Get the function from switcher dictionary
    func = switcher.get(int(args.number), lambda: "Invalid number")
    # Execute the function
    func()