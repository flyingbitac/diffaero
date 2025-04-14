;; Auto-generated. Do not edit!


(when (boundp 'realsense2_camera::IMUInfo)
  (if (not (find-package "REALSENSE2_CAMERA"))
    (make-package "REALSENSE2_CAMERA"))
  (shadow 'IMUInfo (find-package "REALSENSE2_CAMERA")))
(unless (find-package "REALSENSE2_CAMERA::IMUINFO")
  (make-package "REALSENSE2_CAMERA::IMUINFO"))

(in-package "ROS")
;;//! \htmlinclude IMUInfo.msg.html


(defclass realsense2_camera::IMUInfo
  :super ros::object
  :slots (_frame_id _data _noise_variances _bias_variances ))

(defmethod realsense2_camera::IMUInfo
  (:init
   (&key
    ((:frame_id __frame_id) "")
    ((:data __data) (make-array 12 :initial-element 0.0 :element-type :float))
    ((:noise_variances __noise_variances) (make-array 3 :initial-element 0.0 :element-type :float))
    ((:bias_variances __bias_variances) (make-array 3 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _frame_id (string __frame_id))
   (setq _data __data)
   (setq _noise_variances __noise_variances)
   (setq _bias_variances __bias_variances)
   self)
  (:frame_id
   (&optional __frame_id)
   (if __frame_id (setq _frame_id __frame_id)) _frame_id)
  (:data
   (&optional __data)
   (if __data (setq _data __data)) _data)
  (:noise_variances
   (&optional __noise_variances)
   (if __noise_variances (setq _noise_variances __noise_variances)) _noise_variances)
  (:bias_variances
   (&optional __bias_variances)
   (if __bias_variances (setq _bias_variances __bias_variances)) _bias_variances)
  (:serialization-length
   ()
   (+
    ;; string _frame_id
    4 (length _frame_id)
    ;; float64[12] _data
    (* 8    12)
    ;; float64[3] _noise_variances
    (* 8    3)
    ;; float64[3] _bias_variances
    (* 8    3)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _frame_id
       (write-long (length _frame_id) s) (princ _frame_id s)
     ;; float64[12] _data
     (dotimes (i 12)
       (sys::poke (elt _data i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[3] _noise_variances
     (dotimes (i 3)
       (sys::poke (elt _noise_variances i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[3] _bias_variances
     (dotimes (i 3)
       (sys::poke (elt _bias_variances i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _frame_id
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _frame_id (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; float64[12] _data
   (dotimes (i (length _data))
     (setf (elt _data i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64[3] _noise_variances
   (dotimes (i (length _noise_variances))
     (setf (elt _noise_variances i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64[3] _bias_variances
   (dotimes (i (length _bias_variances))
     (setf (elt _bias_variances i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;;
   self)
  )

(setf (get realsense2_camera::IMUInfo :md5sum-) "a02adb3a99530b11ba18a16f40f9512a")
(setf (get realsense2_camera::IMUInfo :datatype-) "realsense2_camera/IMUInfo")
(setf (get realsense2_camera::IMUInfo :definition-)
      "string frame_id
float64[12] data
float64[3] noise_variances
float64[3] bias_variances

")



(provide :realsense2_camera/IMUInfo "a02adb3a99530b11ba18a16f40f9512a")


