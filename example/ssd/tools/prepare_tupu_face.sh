#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python $DIR/prepare_dataset.py --dataset tupu_face --set trainval --target $DIR/../data_face/train.lst --root $DIR/../data_face
python $DIR/prepare_dataset.py --dataset tupu_face --set test --target $DIR/../data_face/val.lst --root $DIR/../data_face --shuffle False
