
(cl:in-package :asdf)

(defsystem "accel_control-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :sensor_msgs-msg
)
  :components ((:file "_package")
    (:file "DepthImage" :depends-on ("_package_DepthImage"))
    (:file "_package_DepthImage" :depends-on ("_package"))
  ))