import threading
import time
import concurrent.futures
import cv2
from threading import Thread
from cv2 import aruco

import numpy as np

from utils import Transform
from utils.Utils import euler_decom
from utils.Calibration import load_calibration_data
from utils.DefaultSettings import aruco_dictionary, aruco_detector_parameters, aruco_cube


class FourDirSensor():
    def __init__(self, verbose: bool = False,
                 calibration_path: str = './calibration',):

        self.default_heading = 0
        self.max_sensing_range = 2.01
        self.cam = cv2.VideoCapture(4)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        self.camera_matrix, self.dist_coeff = load_calibration_data(calibration_path + '/cameraParameters_aruco.xml')
        self.image_size, self.polar_center, self.r_px_length = load_calibration_data(calibration_path + '/cameraParameters_Fisheye.npy')

        self.thread = Thread(target=self.update, args=())
        self.stop_event = threading.Event()
        self.thread.daemon = True
        self.stopped = False
        self.verbose = verbose

        self.h = self.image_size[0]
        self.w = self.image_size[1]
        self.rad_per_px_h = 2 * np.pi / self.w
        self.num_dir = 4
        self.slice_width = int(self.w / self.num_dir)
        self.fisheye_transform = Transform.FisheyeTransform(self.image_size, self.polar_center, self.r_px_length)

        self.dictionary = aruco_dictionary()
        self.parameters = aruco_detector_parameters()
        self.aruco_marker_length = 0.05
        self.dist_coeffs = np.zeros_like(self.dist_coeff)
        self.cubes = [aruco_cube(), aruco_cube([6,7,8,9,10,11])]
        self.quadrant_size = np.array([[[256, 192]]], dtype=np.float32)

    def start(self):
        # start a thread to read frames from the file video stream
        if self.verbose:
            print(f"Start {self.thread.name}: Sensor stream")
            # for dir_id in range(self.num_dir):
            #     cv2.namedWindow(f"dir:{dir_id}")
        self.thread.start()
        return

    def update(self):
        try:
            sample_time = []
            overhead_load = []
            overhead_threads = []
            # while not (self.stopped):
            #     self.start_t = time.time()
            #     ret, frame = self.cam.read()
            #     frame = cv2.cvtColor(self.fisheye_transform.transform(img=frame), cv2.COLOR_BGR2GRAY)
            #     thread_time = time.time()
            #     data, id = self.cube_pose_rel(frame, -1)
            #     end_time_t = time.time()
            #     sample_time.append(end_time_t-self.start_t)
            #     overhead_load.append(thread_time-self.start_t)
            #     overhead_threads.append(end_time_t-thread_time)
            #     # cv2.imshow(f"dir:{0}", frame)
            #     cv2.waitKey(1)
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_dir) as executor:
                while not (self.stopped):
                    self.start_t = time.time()
                    ret, frame = self.cam.read()
                    frame = cv2.cvtColor(self.fisheye_transform.transform(img=frame), cv2.COLOR_BGR2GRAY)
                    future_to_dir_sensor = {executor.submit(self.cube_pose_rel, frame[:, int(dir_id*self.slice_width):int((dir_id+2)*self.slice_width)], dir_id): dir_id for dir_id in range(self.num_dir)}
                    thread_time = time.time()
                    for future in concurrent.futures.as_completed(future_to_dir_sensor):
                        data, id = future.result()
                        print(data, id)
                        # print(frame.shape, img.shape, id, (int(id*self.slice_width), int((id+1)*self.slice_width)))
                        cv2.imshow(f"dir:{id}", frame)
                    cv2.waitKey(1)
                    end_time_t = time.time()
                    sample_time.append(end_time_t-self.start_t)
                    overhead_load.append(thread_time-self.start_t)
                    overhead_threads.append(end_time_t-thread_time)
            print("Sample time", np.mean(sample_time), np.std(sample_time))
            print("Loading time", np.mean(overhead_load), np.std(overhead_load))
            print("Threading time", np.mean(overhead_threads), np.std(overhead_threads))

        except KeyboardInterrupt:
            time.sleep(1)
        except Exception as e:
            if self.verbose:
                print(e)

        cv2.destroyAllWindows()
        self.stopped = True
        return

    def exit(self):
        # indicate that the thread should be stopped
        if self.verbose:
            print("Hub thread is still active:", self.thread.is_alive())
        self.stopped = True
        self.stop_event.set()
        self.thread.join()
        return self.thread.is_alive() and self.stop_event.is_set()

    def cube_pose_rel(self, gray_img, dir_id):
        # detects aruco aruco_tags and returns list of ids and coords of centre
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_img, self.dictionary,
                                                              parameters=self.parameters)
        # corners: tuple([id[corners[x,y]]])
        rel_r = self.max_sensing_range
        heading = self.default_heading
        if ids is not None:
            cornerss = np.array(corners, dtype=np.float32).reshape((len(ids), 4, 2))
            cube_corners = []
            ids_undist = []
            cubes = []
            for cube in self.cubes:
                if ids in cube.ids:
                    ids_cubes_i = np.where(ids == cube.ids)[0]
                    angle_h = np.mean(cornerss[ids_cubes_i, :, 0]) * self.rad_per_px_h
                    rel_phi = angle_h.mean()
                    if np.abs(rel_phi-0.5*np.pi) > 0.25*np.pi:
                        continue
                    else:
                        bearing = rel_phi+0.5*np.pi*dir_id - 0.25*np.pi
                        coord = self.fisheye_transform.spherical2tangent(cornerss[ids_cubes_i])
                        tangent_px_origin = np.mean(cornerss[ids_cubes_i], axis=(0, 1), dtype=np.float32)
                        corners_undist = tuple(coord[ii, np.newaxis, :, :] + tangent_px_origin for ii in range(len(ids_cubes_i)))
                        cube_corners.append(corners_undist)
                        ids_undist.append(ids[ids_cubes_i])
                        cubes.append(cube)

                        _, rvec_undist, tvecs_undist = aruco.estimatePoseBoard(corners_undist, ids[ids_cubes_i], cube,
                                                                               self.camera_matrix,
                                                                               self.dist_coeffs,
                                                                               None,
                                                                               None)
                        rel_cube = np.linalg.norm(np.mean(tvecs_undist))
                        if rel_cube < rel_r:
                            rel_r = rel_cube
                            # polar_coordinates = ((bearing/ np.pi * 180).round(), rel_r.round(4))
                            euler_ang = euler_decom(rvec_undist)
                            heading = (np.mod(bearing + euler_ang[0], 2 * np.pi) / np.pi * 180).round()
                            # print(polar_coordinates[0], polar_coordinates[1], heading, dir_id)
                            cv2.drawFrameAxes(gray_img, self.camera_matrix, self.dist_coeffs, rvec_undist, tvecs_undist, self.aruco_marker_length)
                            aruco.drawDetectedMarkers(gray_img, corners_undist, ids[ids_cubes_i])

            if self.verbose:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners,
                                                                  self.aruco_marker_length,
                                                                  self.camera_matrix,
                                                                  self.dist_coeffs)
                aruco.drawDetectedMarkers(gray_img, corners)
                for i in range(len(ids)):
                    aruco.drawAxis(gray_img, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], self.aruco_marker_length)
                    aruco.drawDetectedMarkers(gray_img, corners)
        data = (heading, rel_r)
        return data, dir_id

    def clean_img(self, img, id):
        img = img[:, int(id * 256):int((id + 1) * 256)]
        return img, id

if __name__ == "__main__":
    test = FourDirSensor(verbose=True)
    test.start()
    time.sleep(30)
    test.exit()
    print("FINISHED")
