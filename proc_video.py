import cv2 as cv
import time
from tqdm import tqdm
import argparse
import pathlib

def process_video(file_name: str , output_file_name: str) -> None:
    """PRocess video file writing its output"""
    cap = cv.VideoCapture(file_name)

    width, height = (
            int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        )
    fps = int(cap.get(cv.CAP_PROP_FPS))
    total_frames = cap.get(7)

    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv.VideoWriter()
    out.open(output_file_name, fourcc, fps, (width, height), False)

    try:
        with tqdm(total=total_frames) as pbar:
            while cap.isOpened():
                pbar.update(1)
                ret, frame = cap.read()
                if not ret:
                    break

                im = frame
                # Example process image
                grey = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
                ########################################################################

                #make inference here & save in out.write(frm)


                ########################################################################
                out.write(grey)
                # cv.imshow("i",grey)
                # if cv.waitKey(1) & 0xFF == ord('q'):
                #     break
        
              
    finally:
        # Release resources
        cap.release()
        out.release()
        $$cv.destroyAllWindows()

    

def single_process(videofile_name: str , output_file_name: str) -> None:
    start_time = time.time()
    process_video(videofile_name, output_file_name)
    end_time = time.time()
    total_processing_time = end_time - start_time
    print("Time taken: {} seconds".format(total_processing_time))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video", help="input file path")
    parser.add_argument("output_file_name", help="output file name")
    args = parser.parse_args()  

    fl = pathlib.Path(args.input_video)
    if not fl.exists:
        print("Input path not valid")
        exit(1)

    videofile_name = args.input_video
    output_file_name = args.output_file_name

    single_process(videofile_name, output_file_name)

if __name__  == "__main__":
    main() 
