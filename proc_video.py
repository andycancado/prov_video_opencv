import cv2 as cv
import time
from tqdm import tqdm
import argparse
import pathlib

def process_img(img: str) -> None:
    tensorflowNet = cv.dnn.readNetFromTensorflow("model/frozen_inference_graph.pb", "model/config.pbtxt")
 
    # Input image
    img = cv.imread(img)
    rows, cols, channels = img.shape
    
    # Use the given image as input, which needs to be blob(s).
    tensorflowNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
    
    # Runs a forward pass to compute the net output
    networkOutput = tensorflowNet.forward()
    
    # Loop on the outputs
    for detection in networkOutput[0,0]:
        
        score = float(detection[2])
        if score > 0.01:
            
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
    
            #draw a red rectangle around detected objects
            cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
    
    # Show the image with a rectagle surrounding the detected objects 
    cv.imshow('Image', img)
    cv.waitKey()
    cv.destroyAllWindows()

def process_video(file_name: str , output_file_name: str) -> None:
    """PRocess video file writing its output"""
    cap = cv.VideoCapture(file_name)
    tensorflowNet = cv.dnn.readNetFromTensorflow("model/frozen_inference_graph.pb", "model/config.pbtxt")


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
                # grey = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
                ########################################################################

                rows, cols, channels = im.shape
                
                # Use the given image as input, which needs to be blob(s).
                tensorflowNet.setInput(cv.dnn.blobFromImage(im, size=(300, 300), swapRB=True, crop=False))
                
                # Runs a forward pass to compute the net output
                networkOutput = tensorflowNet.forward()
                
                # Loop on the outputs
                for detection in networkOutput[0,0]:
                    
                    score = float(detection[2])
                    if score > 0.2:
                        
                        left = detection[3] * cols
                        top = detection[4] * rows
                        right = detection[5] * cols
                        bottom = detection[6] * rows
                
                        #draw a red rectangle around detected objects
                        cv.rectangle(im, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
                
                                #make inference here & save in out.write(frm)


                ########################################################################
                out.write(im)
                cv.imshow("i",im)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
        
              
    finally:
        # Release resources
        cap.release()
        out.release()
        cv.destroyAllWindows()

    

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
    process_img('img.jpg')
    #main() 
