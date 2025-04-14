#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/zxh/ws/wrqws/quaddif/deploy/src/accel_control"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/zxh/ws/wrqws/quaddif/deploy/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/zxh/ws/wrqws/quaddif/deploy/install/lib/python3/dist-packages:/home/zxh/ws/wrqws/quaddif/deploy/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/zxh/ws/wrqws/quaddif/deploy/build" \
    "/home/zxh/miniconda3/envs/ros1/bin/python3" \
    "/home/zxh/ws/wrqws/quaddif/deploy/src/accel_control/setup.py" \
     \
    build --build-base "/home/zxh/ws/wrqws/quaddif/deploy/build/accel_control" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/zxh/ws/wrqws/quaddif/deploy/install" --install-scripts="/home/zxh/ws/wrqws/quaddif/deploy/install/bin"
