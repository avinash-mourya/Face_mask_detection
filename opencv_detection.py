import numpy as np
import tensorflow as tf
import cv2 as cv
import datetime
class Detector():
    def __init__(self,focallength,cap,width):
        self.cap = cap
        self.detection_graph = tf.Graph()
        self.classes ={1:"wearing a mask",2:"not wearing a mask",3: 'mask weared incorrect'}
        self.num_frames=0
        self.focallength=focallength
        self.width=width

    def load_graph(self,fname):
        with self.detection_graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.FastGFile(fname, 'rb') as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')

    def distance_to_camera(self,knownWidth, focalLength, pixelWidth):
        return (knownWidth * focalLength) / pixelWidth

    def calculate_fps(self,start_time):
        self.num_frames += 1
        elapsed_time = (datetime.datetime.now()-start_time).total_seconds()
        fps = self.num_frames / elapsed_time
        return fps
        
    def draw_box(self,img,classId,score,bbox,rows,cols):
        if score > 0.3:
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            confidence ="%.2f"%score
            dist = self.distance_to_camera(self.width,self.focallength,int(right-x))
            dist = "%.2f"%dist
            if classId==1:
                name=self.classes[classId]
                color = (0,255,0)
            elif classId==2:
                name=self.classes[classId]
                color = (0,0,255)
            elif classId==3:
                name=self.classes[classId]
                color = (255,255,0)
            cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)),color, thickness=2)
            cv.putText(img, name, (int(x)-2,int(y)-5), cv.FONT_HERSHEY_COMPLEX, 0.5,color, 1)
            cv.putText(img, f"confidence:{str(confidence)}", (int(x)-2,int(y)-20), cv.FONT_HERSHEY_COMPLEX, 0.5,color, 1)
            cv.putText(img, f"distance from camera:{str(dist)}", (390,440), cv.FONT_HERSHEY_COMPLEX, 0.4,(255,0,0), 1)

    def detect(self,start_time):
        cv.namedWindow("mask detection",cv.WINDOW_NORMAL)
        with tf.Session(graph=self.detection_graph) as sess: 
            while True:
                ret,img = self.cap.read()
                fps = self.calculate_fps(start_time)
                fps = "%.2f"%fps
                rows = img.shape[0]
                cols = img.shape[1]
                inp = cv.resize(img, (224, 224))
                inp = cv.cvtColor(inp,cv.COLOR_BGR2RGB)  # BGR2RGB
                inp = np.expand_dims(inp,axis=0)
                # Run the model
                (num_detections,scores,bboxes,pred_class)= sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                                sess.graph.get_tensor_by_name('detection_scores:0'),
                                sess.graph.get_tensor_by_name('detection_boxes:0'),
                                sess.graph.get_tensor_by_name('detection_classes:0')],
                            feed_dict={'image_tensor:0': inp})

                # Visualize detected bounding boxes.
                for i in range(int(num_detections[0])):
                    classId = int(pred_class[0][i])
                    score = float(scores[0][i])
                    bbox = [float(v) for v in bboxes[0][i]]
                    self.draw_box(img,classId,score,bbox,rows,cols)
                cv.putText(img, f"FPS : {str(fps)}", (20,40), cv.FONT_HERSHEY_COMPLEX, 0.6,(0,255,0), 1)
                cv.imshow("mask detection", img)
                if cv.waitKey(25) & 0xFF == ord('q'):
                    cap.release()
                    cv.destroyAllWindows()
##main
if __name__=='__main__':
    start_time = datetime.datetime.now()
    cap = cv.VideoCapture(0)
    detection = Detector(480,cap,4.0)
    file_n='my_model/frozen_inference_graph.pb'
    detection.load_graph(file_n)
    detection.detect(start_time)








        