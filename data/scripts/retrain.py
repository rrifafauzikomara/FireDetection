r"""Pembelajaran transfer sederhana dengan model Inception v3 atau Mobilenet.

Dengan dukungan untuk TensorBoard.

Contoh ini menunjukkan cara mengambil model Inception v3 atau Mobilenet yang 
dilatih pada gambar ImageNet, dan melatih lapisan atas baru yang dapat mengenali kelas gambar lain.

Lapisan atas menerima sebagai input vektor 2048 dimensi (1001 dimensi untuk Mobilenet) untuk setiap gambar.
Kami melatih lapisan softmax di atas representasi ini. Dengan asumsi lapisan softmax berisi N label, ini sesuai
untuk mempelajari parameter model N + 2048 * N (atau 1001 * N) yang sesuai dengan bias dan bobot yang dipelajari.

Berikut ini merupakan sebuah contoh, yang mengasumsikan Anda memiliki folder yang berisi nama kelas pada
subfolder, masing-masing penuh dengan gambar untuk setiap label. Contoh folder kebakaran
harus memiliki struktur seperti ini:

~/training_data/kebakaran/photo1.jpg
~/training_data/kebakaran/photo2.jpg
...
~/training_data/tidak_kebakaran/anotherphoto77.jpg

Nama-nama subfolder merupakan hal penting, karena mereka menentukan label apa yang diterapkan
pada setiap gambar, tetapi nama file itu sendiri merupakan tidak masalah. Setelah gambar Anda
siap, Anda dapat menjalankan pelatihan dengan perintah seperti ini:


```bash
bazel build tensorflow/examples/image_retraining:retrain && \
bazel-bin/tensorflow/examples/image_retraining/retrain \
    --image_dir ~/training_data
```

Atau, jika Anda memiliki instalasi pip tensorflow, `retrain.py` Anda dapat 
menjalankannya tanpa bazel:

```bash
python tensorflow/examples/image_retraining/retrain.py \
    --image_dir ~/training_data
```

Anda dapat mengganti argumen image_dir dengan folder apa pun yang berisi subfolder dari
gambar. Label untuk setiap gambar diambil dari nama subfoldernya.

Ini menghasilkan file model baru yang dapat dimuat dan dijalankan oleh program TensorFlow apa pun, 
misalnya kode sampel label_image.

Secara default skrip ini akan menggunakan akurasi tinggi, tetapi relatif besar dan
arsitektur model Inception v3 lambat. Disarankan agar Anda mulai dengan ini untuk memvalidasi 
bahwa Anda telah mengumpulkan data pelatihan yang baik, tetapi jika Anda ingin menggunakan
pada platform terbatas sumber daya, Anda dapat mencoba tanda `--architecture` dengan sebuah
Model mobilenet. Sebagai contoh:

```bash
python tensorflow/examples/image_retraining/retrain.py \
    --image_dir ~/training_data --architecture mobilenet_1.0_224
```

Ada 32 model Mobilenet yang berbeda untuk dipilih, dengan beragam file
opsi ukuran dan latensi. Angka pertama bisa '1,0', '0,75', '0,50', atau
'0,25' untuk mengontrol ukuran, dan yang kedua mengontrol ukuran gambar input
'224', '192', '160', atau '128', dengan ukuran lebih kecil berjalan lebih cepat. Lihat
https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html
untuk informasi lebih lanjut tentang Mobilenet.

Untuk digunakan dengan TensorBoard:

Secara default, skrip ini akan mencatat ringkasan ke direktori / tmp / retrain_logs

Visualisasikan ringkasan dengan perintah ini:

tensorboard --logdir /tmp/retrain_logs

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

FLAGS = None

# Ini semua adalah parameter yang terkait dengan arsitektur model tertentu
# kami menggunakan untuk Inception v3. Ini termasuk hal-hal seperti nama tensor dan nama mereka
# Jika Anda ingin mengadaptasi skrip ini agar berfungsi dengan model lain, Anda akan melakukannya
# perlu memperbarui ini untuk mencerminkan nilai-nilai di jaringan yang Anda gunakan.
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def create_image_lists(image_dir, testing_percentage, validation_percentage):
  """Membuat daftar gambar pelatihan dari sistem file.

   Menganalisis sub folder dalam direktori gambar, membaginya menjadi stabil
   set pelatihan, pengujian, dan validasi, dan mengembalikan struktur data
   menjelaskan daftar gambar untuk setiap label dan jalurnya.

   Args:
     image_dir: String path ke folder yang berisi subfolder gambar.
     testing_percentage: Persentase integral dari gambar untuk dicadangkan untuk pengujian.
     validation_percentage: Persentase integer dari gambar yang disediakan untuk validasi.

   Return:
     Kamus yang berisi entri untuk setiap subfolder label, dengan gambar terpecah
     ke dalam set pelatihan, pengujian, dan validasi dalam setiap label.
   """
  if not gfile.Exists(image_dir):
    tf.logging.error("Image directory '" + image_dir + "' not found.")
    return None
  result = collections.OrderedDict()
  sub_dirs = [
    os.path.join(image_dir,item)
    for item in gfile.ListDirectory(image_dir)]
  sub_dirs = sorted(item for item in sub_dirs
                    if gfile.IsDirectory(item))
  for sub_dir in sub_dirs:
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []
    dir_name = os.path.basename(sub_dir)
    if dir_name == image_dir:
      continue
    tf.logging.info("Looking for images in '" + dir_name + "'")
    for extension in extensions:
      file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
      file_list.extend(gfile.Glob(file_glob))
    if not file_list:
      tf.logging.warning('No files found')
      continue
    if len(file_list) < 20:
      tf.logging.warning(
          'WARNING: Folder has less than 20 images, which may cause issues.')
    elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
      tf.logging.warning(
          'WARNING: Folder {} has more than {} images. Some images will '
          'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
    label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    training_images = []
    testing_images = []
    validation_images = []
    for file_name in file_list:
      base_name = os.path.basename(file_name)
      # We want to ignore anything after '_nohash_' in the file name when
      # deciding which set to put an image in, the data set creator has a way of
      # grouping photos that are close variations of each other. For example
      # this is used in the plant disease data set to group multiple pictures of
      # the same leaf.
      hash_name = re.sub(r'_nohash_.*$', '', file_name)
      # This looks a bit magical, but we need to decide whether this file should
      # go into the training, testing, or validation sets, and we want to keep
      # existing files in the same set even if more files are subsequently
      # added.
      # To do that, we need a stable way of deciding based on just the file name
      # itself, so we do a hash of that and then use that to generate a
      # probability value that we use to assign it.
      hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
      percentage_hash = ((int(hash_name_hashed, 16) %
                          (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                         (100.0 / MAX_NUM_IMAGES_PER_CLASS))
      if percentage_hash < validation_percentage:
        validation_images.append(base_name)
      elif percentage_hash < (testing_percentage + validation_percentage):
        testing_images.append(base_name)
      else:
        training_images.append(base_name)
    result[label_name] = {
        'dir': dir_name,
        'training': training_images,
        'testing': testing_images,
        'validation': validation_images,
    }
  return result


def get_image_path(image_lists, label_name, index, image_dir, category):
  """ Mengembalikan path ke gambar untuk label di indeks yang diberikan.

   Args:
     image_lists: Kamus gambar latihan untuk setiap label.
     label_name: Label string yang ingin kita dapatkan gambarnya.
     index: Int offset dari gambar yang kita inginkan. Ini akan dimodulasi oleh
     tersedia sejumlah gambar untuk label, sehingga dapat menjadi besar secara sewenang-wenang.
     image_dir: Root string folder dari subfolder yang berisi pelatihan
     gambar.
     category: Nama string yang diatur untuk menarik gambar dari - pelatihan, pengujian, atau
     validasi.

   Return:
     File jalur jalur sistem ke gambar yang memenuhi parameter yang diminta.

   """
  if label_name not in image_lists:
    tf.logging.fatal('Label does not exist %s.', label_name)
  label_lists = image_lists[label_name]
  if category not in label_lists:
    tf.logging.fatal('Category does not exist %s.', category)
  category_list = label_lists[category]
  if not category_list:
    tf.logging.fatal('Label %s has no images in the category %s.',
                     label_name, category)
  mod_index = index % len(category_list)
  base_name = category_list[mod_index]
  sub_dir = label_lists['dir']
  full_path = os.path.join(image_dir, sub_dir, base_name)
  return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
                        category, architecture):
  """ Mengembalikan jalur ke file bottleneck untuk label di indeks yang diberikan.

   Args:
     image_lists: Kamus gambar latihan untuk setiap label.
     label_name: Label string yang ingin kita dapatkan gambarnya.
     index: Integer offset dari gambar yang kita inginkan. Ini akan dimodulasi oleh
     tersedia sejumlah gambar untuk label, sehingga dapat menjadi besar secara sewenang-wenang.
     bottleneck_dir: String folder yang menyimpan file cache dari nilai bottleneck.
     category: Nama string yang diatur untuk menarik gambar dari - pelatihan, pengujian, atau
     validasi.
     architecture: Nama model arsitektur.

   Return:
     File jalur jalur sistem ke gambar yang memenuhi parameter yang diminta.
   """
  return get_image_path(image_lists, label_name, index, bottleneck_dir,
                        category) + '_' + architecture + '.txt'


def create_model_graph(model_info):
  """Membuat grafik dari file GraphDef yang disimpan dan mengembalikan objek Graph.

   Args:
     model_info: Kamus yang berisi informasi tentang arsitektur model.

   Return:
     Grafik memegang jaringan Inception V3 yangterlatih, dan berbagai tensor kita akan
     memanipulasi.
   """
  with tf.Graph().as_default() as graph:
    model_path = os.path.join(FLAGS.model_dir, model_info['model_file_name'])
    with gfile.FastGFile(model_path, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
          graph_def,
          name='',
          return_elements=[
              model_info['bottleneck_tensor_name'],
              model_info['resized_input_tensor_name'],
          ]))
  return graph, bottleneck_tensor, resized_input_tensor


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            decoded_image_tensor, resized_input_tensor,
                            bottleneck_tensor):
  """Menjalankan inferensi pada gambar untuk mengekstrak lapisan ringkasan 'bottleneck'.

   Args:
     sess: Sesi TensorFlow aktif saat ini.
     image_data: String data JPEG mentah.
     image_data_tensor: Input layer data dalam grafik.
     decoded_image_tensor: Output dari pengubahan ukuran dan preprocess gambar awal.
     resized_input_tensor: Node input dari grafik pengenalan.
     bottleneck_tensor: Lapisan sebelum softmax akhir.

   Return:
     Array numpy dari nilai bottleneck.
   """
  # First decode the JPEG image, resize it, and rescale the pixel values.
  resized_input_values = sess.run(decoded_image_tensor,
                                  {image_data_tensor: image_data})
  # Then run it through the recognition network.
  bottleneck_values = sess.run(bottleneck_tensor,
                               {resized_input_tensor: resized_input_values})
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values


def maybe_download_and_extract(data_url):
  """ Download dan ekstrak file tar model.

   Jika model pra-latihan yang digunakan belum ada, fungsi ini akan
   mengunduhnya dari situs web TensorFlow.org dan membukanya di direktori.

   Args:
     data_url: Lokasi web dari file tar yang berisi model pra-pelatihan.
   """
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = data_url.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):

    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' %
                       (filename,
                        float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    tf.logging.info('Successfully downloaded', filename, statinfo.st_size,
                    'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def ensure_dir_exists(dir_name):
  """Makes sure the folder exists on disk.

  Args:
    dir_name: Path string to the folder we want to create.
  """
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)


bottleneck_path_2_bottleneck_values = {}


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor):
  """Create a single bottleneck file."""
  tf.logging.info('Creating bottleneck at ' + bottleneck_path)
  image_path = get_image_path(image_lists, label_name, index,
                              image_dir, category)
  if not gfile.Exists(image_path):
    tf.logging.fatal('File does not exist %s', image_path)
  image_data = gfile.FastGFile(image_path, 'rb').read()
  try:
    bottleneck_values = run_bottleneck_on_image(
        sess, image_data, jpeg_data_tensor, decoded_image_tensor,
        resized_input_tensor, bottleneck_tensor)
  except Exception as e:
    raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                 str(e)))
  bottleneck_string = ','.join(str(x) for x in bottleneck_values)
  with open(bottleneck_path, 'w') as bottleneck_file:
    bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             decoded_image_tensor, resized_input_tensor,
                             bottleneck_tensor, architecture):
  """Mengambil atau menghitung nilai hambatan untuk gambar.

  Jika versi cache dari data bottleneck ada di-disk, kembalikan itu,
  jika tidak, hitung data dan simpan ke disk untuk digunakan di masa mendatang.

  Args:
    sess: Sesi TensorFlow aktif saat ini.
    image_lists: Kamus gambar latihan untuk setiap label.
    label_name: Label string yang ingin kita dapatkan gambarnya.
    index: Integer offset dari gambar yang kita inginkan. Ini akan dimodulasi oleh
    tersedia sejumlah gambar untuk label, sehingga dapat menjadi besar secara sewenang-wenang.
    image_dir: Root string folder dari subfolder yang berisi pelatihan
    gambar.
    kategori: Nama string yang ditetapkan untuk menarik gambar dari - pelatihan, pengujian,
    atau validasi.
    bottleneck_dir: String folder yang menyimpan file cache dari nilai bottleneck.
    jpeg_data_tensor: tensor untuk memasukkan data jpeg yang dimuat ke.
    decoded_image_tensor: Output dari decoding dan mengubah ukuran gambar.
    resized_input_tensor: Node input dari grafik pengenalan.
    bottleneck_tensor: tensor output untuk nilai bottleneck.
    architecture: Nama arsitektur model.

  Return:
    Array numpy nilai yang dihasilkan oleh lapisan bottleneck untuk gambar.
  """
  label_lists = image_lists[label_name]
  sub_dir = label_lists['dir']
  sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
  ensure_dir_exists(sub_dir_path)
  bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                        bottleneck_dir, category, architecture)
  if not os.path.exists(bottleneck_path):
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor)
  with open(bottleneck_path, 'r') as bottleneck_file:
    bottleneck_string = bottleneck_file.read()
  did_hit_error = False
  try:
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  except ValueError:
    tf.logging.warning('Invalid float found, recreating bottleneck')
    did_hit_error = True
  if did_hit_error:
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
      bottleneck_string = bottleneck_file.read()
    # Allow exceptions to propagate here, since they shouldn't happen after a
    # fresh creation
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor, architecture):
  """Memastikan semua hambatan pelatihan, pengujian, dan validasi di-cache.

  Karena kita cenderung membaca gambar yang sama beberapa kali (jika tidak ada
  distorsi diterapkan selama pelatihan) itu dapat mempercepat banyak hal jika kita
  menghitung nilai bottleneck layer sekali untuk setiap gambar selama
  preprocessing, dan kemudian hanya membaca nilai-nilai yang di-cache berulang kali selama
  latihan. Di sini kita melihat semua gambar yang kami temukan, menghitungnya
  nilai, dan simpan.

  Args:
    sess: Sesi TensorFlow aktif saat ini.
    image_lists: Kamus gambar latihan untuk setiap label.
    image_dir: Root string folder dari subfolder yang berisi pelatihan
    gambar.
    bottleneck_dir: String folder yang menyimpan file cache dari nilai bottleneck.
    jpeg_data_tensor: Input tensor untuk data jpeg dari file.
    decoded_image_tensor: Output dari decoding dan mengubah ukuran gambar.
    resized_input_tensor: Node input dari grafik pengenalan.
    bottleneck_tensor: Lapisan keluaran kedua dari grafik.
    architecture: Nama arsitektur model.

  Return:
    Tidak ada.
  """
  how_many_bottlenecks = 0
  ensure_dir_exists(bottleneck_dir)
  for label_name, label_lists in image_lists.items():
    for category in ['training', 'testing', 'validation']:
      category_list = label_lists[category]
      for index, unused_base_name in enumerate(category_list):
        get_or_create_bottleneck(
            sess, image_lists, label_name, index, image_dir, category,
            bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor, architecture)

        how_many_bottlenecks += 1
        if how_many_bottlenecks % 100 == 0:
          tf.logging.info(
              str(how_many_bottlenecks) + ' bottleneck files created.')


def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  decoded_image_tensor, resized_input_tensor,
                                  bottleneck_tensor, architecture):
  """Mengambil nilai bottleneck untuk gambar yang di-cache.

  Jika tidak ada distorsi yang diterapkan, fungsi ini dapat mengambil cache
  nilai bottleneck langsung dari disk untuk gambar. Ini mengambil set acak
  gambar dari kategori yang ditentukan.

  Args:
    sess: Sesi TensorFlow Saat Ini.
    image_lists: Kamus gambar latihan untuk setiap label.
    how_many: Jika positif, sampel acak dari ukuran ini akan dipilih.
    Jika negatif, semua kemacetan akan diambil.
    category: Nama string yang ditetapkan untuk menarik dari - pelatihan, pengujian, atau validasi.
    bottleneck_dir: String folder yang menyimpan file cache dari nilai bottleneck.
    image_dir: Root string folder dari subfolder yang berisi pelatihan gambar.
    jpeg_data_tensor: Lapisan untuk memasukkan data gambar jpeg ke.
    decoded_image_tensor: Output dari decoding dan mengubah ukuran gambar.
    resized_input_tensor: Node input dari grafik pengenalan.
    bottleneck_tensor: Lapisan keluaran bottleneck dari grafik CNN.
    architecture: Nama arsitektur model.

  Pengembalian:
    Daftar array kemacetan, kebenaran dasar yang sesuai, dan
    nama file yang relevan.
  """
  class_count = len(image_lists.keys())
  bottlenecks = []
  ground_truths = []
  filenames = []
  if how_many >= 0:
    # Retrieve a random sample of bottlenecks.
    for unused_i in range(how_many):
      label_index = random.randrange(class_count)
      label_name = list(image_lists.keys())[label_index]
      image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
      image_name = get_image_path(image_lists, label_name, image_index,
                                  image_dir, category)
      bottleneck = get_or_create_bottleneck(
          sess, image_lists, label_name, image_index, image_dir, category,
          bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
          resized_input_tensor, bottleneck_tensor, architecture)
      ground_truth = np.zeros(class_count, dtype=np.float32)
      ground_truth[label_index] = 1.0
      bottlenecks.append(bottleneck)
      ground_truths.append(ground_truth)
      filenames.append(image_name)
  else:
    # Retrieve all bottlenecks.
    for label_index, label_name in enumerate(image_lists.keys()):
      for image_index, image_name in enumerate(
          image_lists[label_name][category]):
        image_name = get_image_path(image_lists, label_name, image_index,
                                    image_dir, category)
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, image_dir, category,
            bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor, architecture)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
        filenames.append(image_name)
  return bottlenecks, ground_truths, filenames


def get_random_distorted_bottlenecks(
    sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,
    distorted_image, resized_input_tensor, bottleneck_tensor):
  """Mengambil nilai bottleneck untuk melatih gambar, setelah distorsi.

  Jika kita berlatih dengan distorsi seperti tanaman, timbangan, atau membalik, kita harus melakukannya
  hitung ulang model lengkap untuk setiap gambar, jadi kami tidak dapat menggunakan cache
  nilai bottleneck. Alih-alih kami menemukan gambar acak untuk kategori yang diminta,
  jalankan mereka melalui grafik distorsi, dan kemudian grafik lengkap untuk mendapatkan
  hasil bottleneck untuk masing-masing.

  Args:
    sess: Sesi TensorFlow Saat Ini.
    image_lists: Kamus gambar latihan untuk setiap label.
    how_many: Jumlah integer dari nilai bottleneck untuk dikembalikan.
    category: String nama yang kumpulan gambarnya akan diambil - pelatihan, pengujian, atau validasi.
    image_dir: Root string folder dari subfolder yang berisi pelatihan gambar.
    input_jpeg_tensor: Lapisan input tempat kami mengumpankan data gambar.
    distorted_image: Node output dari grafik distorsi.
    resized_input_tensor: Node input dari grafik pengenalan.
    bottleneck_tensor: Lapisan keluaran bottleneck dari grafik CNN.

  Pengembalian:
    Daftar array kemacetan dan kebenaran dasarnya.
  """
  class_count = len(image_lists.keys())
  bottlenecks = []
  ground_truths = []
  for unused_i in range(how_many):
    label_index = random.randrange(class_count)
    label_name = list(image_lists.keys())[label_index]
    image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
    image_path = get_image_path(image_lists, label_name, image_index, image_dir,
                                category)
    if not gfile.Exists(image_path):
      tf.logging.fatal('File does not exist %s', image_path)
    jpeg_data = gfile.FastGFile(image_path, 'rb').read()
    # Note that we materialize the distorted_image_data as a numpy array before
    # sending running inference on the image. This involves 2 memory copies and
    # might be optimized in other implementations.
    distorted_image_data = sess.run(distorted_image,
                                    {input_jpeg_tensor: jpeg_data})
    bottleneck_values = sess.run(bottleneck_tensor,
                                 {resized_input_tensor: distorted_image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    ground_truth = np.zeros(class_count, dtype=np.float32)
    ground_truth[label_index] = 1.0
    bottlenecks.append(bottleneck_values)
    ground_truths.append(ground_truth)
  return bottlenecks, ground_truths


def should_distort_images(flip_left_right, random_crop, random_scale,
                          random_brightness):
  """Apakah ada distorsi yang diaktifkan, dari flag input.

   Args:
     flip_left_right: Boolean apakah akan secara acak mencerminkan gambar secara horizontal.
     random_crop: Persentase integer yang mengatur margin total yang digunakan di sekitar kotak krop.
     random_scale: Persentase integer berapa banyak variasi skala menurut.
     random_brightness: Kisaran integer untuk secara acak mengalikan nilai piksel dengan.

   Pengembalian:
     Nilai Boolean yang menunjukkan apakah distorsi harus diterapkan.
  """
  return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
          (random_brightness != 0))


def add_input_distortions(flip_left_right, random_crop, random_scale,
                          random_brightness, input_width, input_height,
                          input_depth, input_mean, input_std):
  """Membuat operasi untuk menerapkan distorsi yang ditentukan.

   Selama pelatihan itu dapat membantu meningkatkan hasil jika kita menjalankan gambar
   melalui distorsi sederhana seperti tanaman, sisik, dan membalik. Ini mencerminkan
   jenis variasi yang kita harapkan di dunia nyata, dan dengan demikian dapat membantu melatihnya
   model untuk mengatasi data alami lebih efektif. Di sini kami mengambil yang disediakan
   parameter dan membangun jaringan operasi untuk menerapkannya pada gambar.

  Cropping
  ~~~~~~~~

  Pemangkasan dilakukan dengan menempatkan kotak pembatas pada posisi acak penuh
  gambar. Parameter pemotongan mengontrol ukuran kotak itu relatif terhadap
  masukan gambar. Jika nol, maka kotak itu berukuran sama dengan input dan tidak
  tanam dilakukan. Jika nilainya 50%, maka kotak potong akan menjadi setengahnya
  lebar dan tinggi input. Dalam diagram terlihat seperti ini:

  <       width         >
  +---------------------+
  |                     |
  |   width - crop%     |
  |    <      >         |
  |    +------+         |
  |    |      |         |
  |    |      |         |
  |    |      |         |
  |    +------+         |
  |                     |
  |                     |
  +---------------------+

  Scaling
  ~~~~~~~

  Penskalaan sangat mirip memotong, kecuali bahwa kotak pembatas selalu
  terpusat dan ukurannya bervariasi secara acak dalam kisaran yang diberikan. Misalnya jika
  persentase skala adalah nol, maka kotak pembatas adalah ukuran yang sama dengan
  input dan tidak ada skala yang diterapkan. Jika 50%, maka kotak pembatas akan berada di
  rentang acak antara setengah lebar dan tinggi dan ukuran penuh.

  Args:
    flip_left_right: Boolean apakah akan secara acak mencerminkan gambar secara horizontal.
    random_crop: Persentase integer yang mengatur margin total yang digunakan di sekitar
    kotak krop.
    random_scale: Persentase integer berapa banyak variasi skala menurut.
    random_brightness: Kisaran integer untuk secara acak mengalikan nilai piksel dengan.
    grafik.
    input_width: Ukuran horisontal dari gambar input yang diharapkan untuk dimodelkan.
    input_height: Ukuran vertikal dari gambar input yang diharapkan untuk dimodelkan.
    input_depth: Berapa banyak saluran yang seharusnya dimiliki gambar input.
    input_mean: Nilai piksel yang seharusnya nol pada gambar untuk grafik.
    input_std: Berapa banyak untuk membagi nilai pixel sebelum pengenalan.

  Pengembalian:
    Lapisan input jpeg dan tensor hasil terdistorsi.
  """

  jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
  decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  margin_scale = 1.0 + (random_crop / 100.0)
  resize_scale = 1.0 + (random_scale / 100.0)
  margin_scale_value = tf.constant(margin_scale)
  resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
                                         minval=1.0,
                                         maxval=resize_scale)
  scale_value = tf.multiply(margin_scale_value, resize_scale_value)
  precrop_width = tf.multiply(scale_value, input_width)
  precrop_height = tf.multiply(scale_value, input_height)
  precrop_shape = tf.stack([precrop_height, precrop_width])
  precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
  precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                              precrop_shape_as_int)
  precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
  cropped_image = tf.random_crop(precropped_image_3d,
                                 [input_height, input_width, input_depth])
  if flip_left_right:
    flipped_image = tf.image.random_flip_left_right(cropped_image)
  else:
    flipped_image = cropped_image
  brightness_min = 1.0 - (random_brightness / 100.0)
  brightness_max = 1.0 + (random_brightness / 100.0)
  brightness_value = tf.random_uniform(tensor_shape.scalar(),
                                       minval=brightness_min,
                                       maxval=brightness_max)
  brightened_image = tf.multiply(flipped_image, brightness_value)
  offset_image = tf.subtract(brightened_image, input_mean)
  mul_image = tf.multiply(offset_image, 1.0 / input_std)
  distort_result = tf.expand_dims(mul_image, 0, name='DistortResult')
  return jpeg_data, distort_result


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor,
                           bottleneck_tensor_size):
  """Menambahkan softmax baru dan lapisan yang sepenuhnya terhubung untuk pelatihan.

   Kita perlu melatih ulang lapisan atas untuk mengidentifikasi kelas baru kita, jadi ini berfungsi
   menambahkan operasi yang tepat ke grafik, bersama dengan beberapa variabel untuk menahan
   bobot, dan kemudian mengatur semua gradien untuk umpan mundur.

   Pengaturan untuk softmax dan lapisan yang sepenuhnya terhubung didasarkan pada:
   https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

   Args:
     class_count: Integer dari berapa banyak kategori hal yang kami coba
     mengakui.
     final_tensor_name: Nama string untuk simpul akhir baru yang menghasilkan hasil.
     bottleneck_tensor: Output dari grafik CNN utama.
     bottleneck_tensor_size: Berapa banyak entri dalam vektor bottleneck.

   Pengembalian:
     Tensor untuk pelatihan dan hasil cross entropy, dan tensor untuk
     input bottleneck dan input ground truth.
  """
  with tf.name_scope('input'):
    bottleneck_input = tf.placeholder_with_default(
        bottleneck_tensor,
        shape=[None, bottleneck_tensor_size],
        name='BottleneckInputPlaceholder')

    ground_truth_input = tf.placeholder(tf.float32,
                                        [None, class_count],
                                        name='GroundTruthInput')

  # Organizing the following ops as `final_training_ops` so they're easier
  # to see in TensorBoard
  layer_name = 'final_training_ops'
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      initial_value = tf.truncated_normal(
          [bottleneck_tensor_size, class_count], stddev=0.001)

      layer_weights = tf.Variable(initial_value, name='final_weights')

      variable_summaries(layer_weights)
    with tf.name_scope('biases'):
      layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
      variable_summaries(layer_biases)
    with tf.name_scope('Wx_plus_b'):
      logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
      tf.summary.histogram('pre_activations', logits)

  final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
  tf.summary.histogram('activations', final_tensor)

  with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=ground_truth_input, logits=logits)
    with tf.name_scope('total'):
      cross_entropy_mean = tf.reduce_mean(cross_entropy)
  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    train_step = optimizer.minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
          final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
  """Menyisipkan operasi yang kami butuhkan untuk mengevaluasi akurasi hasil kami.

   Args:
     result_tensor: Node akhir baru yang menghasilkan hasil.
     ground_truth_tensor: Node yang kami beri makan data ground kebenaran
     ke.

   Pengembalian:
     Tuple of (langkah evaluasi, prediksi).
  """
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      prediction = tf.argmax(result_tensor, 1)
      correct_prediction = tf.equal(
          prediction, tf.argmax(ground_truth_tensor, 1))
    with tf.name_scope('accuracy'):
      evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', evaluation_step)
  return evaluation_step, prediction


def save_graph_to_file(sess, graph, graph_file_name):
  output_graph_def = graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
  with gfile.FastGFile(graph_file_name, 'wb') as f:
    f.write(output_graph_def.SerializeToString())
  return


def prepare_file_system():
  # Setup the directory we'll write summaries to for TensorBoard
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)
  if FLAGS.intermediate_store_frequency > 0:
    ensure_dir_exists(FLAGS.intermediate_output_graphs_dir)
  return


def create_model_info(architecture):
  """Diberi nama arsitektur model, mengembalikan informasi tentang itu.

   Ada berbagai model dasar yang dapat dikenali oleh pengenalan citra dasar yang
   dilatih ulang menggunakan transfer belajar, dan fungsi ini diterjemahkan dari namanya
   dari model dengan atribut yang diperlukan untuk mengunduh dan berlatih dengannya.

   Args:
     architecture: Nama arsitektur model.

   Pengembalian:
     Kamus informasi tentang model, atau Tidak ada jika namanya tidak
     diakui

   Meningkatkan:
     ValueError: Jika nama arsitektur tidak dikenal.
  """
  architecture = architecture.lower()
  if architecture == 'inception_v3':
    # pylint: disable=line-too-long
    data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    # pylint: enable=line-too-long
    bottleneck_tensor_name = 'pool_3/_reshape:0'
    bottleneck_tensor_size = 2048
    input_width = 299
    input_height = 299
    input_depth = 3
    resized_input_tensor_name = 'Mul:0'
    model_file_name = 'classify_image_graph_def.pb'
    input_mean = 128
    input_std = 128
  elif architecture.startswith('mobilenet_'):
    parts = architecture.split('_')
    if len(parts) != 3 and len(parts) != 4:
      tf.logging.error("Couldn't understand architecture name '%s'",
                       architecture)
      return None
    version_string = parts[1]
    if (version_string != '1.0' and version_string != '0.75' and
        version_string != '0.50' and version_string != '0.25'):
      tf.logging.error(
          """"The Mobilenet version should be '1.0', '0.75', '0.50', or '0.25',
  but found '%s' for architecture '%s'""",
          version_string, architecture)
      return None
    size_string = parts[2]
    if (size_string != '224' and size_string != '192' and
        size_string != '160' and size_string != '128'):
      tf.logging.error(
          """The Mobilenet input size should be '224', '192', '160', or '128',
 but found '%s' for architecture '%s'""",
          size_string, architecture)
      return None
    if len(parts) == 3:
      is_quantized = False
    else:
      if parts[3] != 'quantized':
        tf.logging.error(
            "Couldn't understand architecture suffix '%s' for '%s'", parts[3],
            architecture)
        return None
      is_quantized = True
    data_url = 'http://download.tensorflow.org/models/mobilenet_v1_'
    data_url += version_string + '_' + size_string + '_frozen.tgz'
    bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
    bottleneck_tensor_size = 1001
    input_width = int(size_string)
    input_height = int(size_string)
    input_depth = 3
    resized_input_tensor_name = 'input:0'
    if is_quantized:
      model_base_name = 'quantized_graph.pb'
    else:
      model_base_name = 'frozen_graph.pb'
    model_dir_name = 'mobilenet_v1_' + version_string + '_' + size_string
    model_file_name = os.path.join(model_dir_name, model_base_name)
    input_mean = 127.5
    input_std = 127.5
  else:
    tf.logging.error("Couldn't understand architecture name '%s'", architecture)
    raise ValueError('Unknown architecture', architecture)

  return {
      'data_url': data_url,
      'bottleneck_tensor_name': bottleneck_tensor_name,
      'bottleneck_tensor_size': bottleneck_tensor_size,
      'input_width': input_width,
      'input_height': input_height,
      'input_depth': input_depth,
      'resized_input_tensor_name': resized_input_tensor_name,
      'model_file_name': model_file_name,
      'input_mean': input_mean,
      'input_std': input_std,
  }


def add_jpeg_decoding(input_width, input_height, input_depth, input_mean,
                      input_std):
  """Menambahkan operasi yang melakukan decoding dan mengubah ukuran JPEG ke grafik ..

   Args:
     input_width: Lebar gambar yang diinginkan dimasukkan ke dalam grafik pengenal.
     input_height: Lebar gambar yang diinginkan dimasukkan ke dalam grafik pengenal.
     input_depth: Saluran gambar yang diinginkan dimasukkan ke dalam grafik pengenal.
     input_mean: Nilai piksel yang seharusnya nol pada gambar untuk grafik.
     input_std: Berapa banyak untuk membagi nilai pixel sebelum pengenalan.

   Pengembalian:
     Tensor untuk node untuk memasukkan data JPEG ke dalam, dan output dari
       langkah preprocessing.
  """
  jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
  decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  resize_shape = tf.stack([input_height, input_width])
  resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
  resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                           resize_shape_as_int)
  offset_image = tf.subtract(resized_image, input_mean)
  mul_image = tf.multiply(offset_image, 1.0 / input_std)
  return jpeg_data, mul_image


def main(_):
  # Needed to make sure the logging output is visible.
  # See https://github.com/tensorflow/tensorflow/issues/3047
  tf.logging.set_verbosity(tf.logging.INFO)

  # Prepare necessary directories  that can be used during training
  prepare_file_system()

  # Gather information about the model architecture we'll be using.
  model_info = create_model_info(FLAGS.architecture)
  if not model_info:
    tf.logging.error('Did not recognize architecture flag')
    return -1

  # Set up the pre-trained graph.
  maybe_download_and_extract(model_info['data_url'])
  graph, bottleneck_tensor, resized_image_tensor = (
      create_model_graph(model_info))

  # Look at the folder structure, and create lists of all the images.
  image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
                                   FLAGS.validation_percentage)
  class_count = len(image_lists.keys())
  if class_count == 0:
    tf.logging.error('No valid folders of images found at ' + FLAGS.image_dir)
    return -1
  if class_count == 1:
    tf.logging.error('Only one valid folder of images found at ' +
                     FLAGS.image_dir +
                     ' - multiple classes are needed for classification.')
    return -1

  # See if the command-line flags mean we're applying any distortions.
  do_distort_images = should_distort_images(
      FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
      FLAGS.random_brightness)

  with tf.Session(graph=graph) as sess:
    # Set up the image decoding sub-graph.
    jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
        model_info['input_width'], model_info['input_height'],
        model_info['input_depth'], model_info['input_mean'],
        model_info['input_std'])

    if do_distort_images:
      # We will be applying distortions, so setup the operations we'll need.
      (distorted_jpeg_data_tensor,
       distorted_image_tensor) = add_input_distortions(
           FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
           FLAGS.random_brightness, model_info['input_width'],
           model_info['input_height'], model_info['input_depth'],
           model_info['input_mean'], model_info['input_std'])
    else:
      # We'll make sure we've calculated the 'bottleneck' image summaries and
      # cached them on disk.
      cache_bottlenecks(sess, image_lists, FLAGS.image_dir,
                        FLAGS.bottleneck_dir, jpeg_data_tensor,
                        decoded_image_tensor, resized_image_tensor,
                        bottleneck_tensor, FLAGS.architecture)

    # Add the new layer that we'll be training.
    (train_step, cross_entropy, bottleneck_input, ground_truth_input,
     final_tensor) = add_final_training_ops(
         len(image_lists.keys()), FLAGS.final_tensor_name, bottleneck_tensor,
         model_info['bottleneck_tensor_size'])

    # Create the operations we need to evaluate the accuracy of our new layer.
    evaluation_step, prediction = add_evaluation_step(
        final_tensor, ground_truth_input)

    # Merge all the summaries and write them out to the summaries_dir
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)

    validation_writer = tf.summary.FileWriter(
        FLAGS.summaries_dir + '/validation')

    # Set up all our weights to their initial default values.
    init = tf.global_variables_initializer()
    sess.run(init)

    # Run the training for as many cycles as requested on the command line.
    for i in range(FLAGS.how_many_training_steps):
      # Get a batch of input bottleneck values, either calculated fresh every
      # time with distortions applied, or from the cache stored on disk.
      if do_distort_images:
        (train_bottlenecks,
         train_ground_truth) = get_random_distorted_bottlenecks(
             sess, image_lists, FLAGS.train_batch_size, 'training',
             FLAGS.image_dir, distorted_jpeg_data_tensor,
             distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
      else:
        (train_bottlenecks,
         train_ground_truth, _) = get_random_cached_bottlenecks(
             sess, image_lists, FLAGS.train_batch_size, 'training',
             FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
             decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
             FLAGS.architecture)
      # Feed the bottlenecks and ground truth into the graph, and run a training
      # step. Capture training summaries for TensorBoard with the `merged` op.
      train_summary, _ = sess.run(
          [merged, train_step],
          feed_dict={bottleneck_input: train_bottlenecks,
                     ground_truth_input: train_ground_truth})
      train_writer.add_summary(train_summary, i)

      # Every so often, print out how well the graph is training.
      is_last_step = (i + 1 == FLAGS.how_many_training_steps)
      if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
        train_accuracy, cross_entropy_value = sess.run(
            [evaluation_step, cross_entropy],
            feed_dict={bottleneck_input: train_bottlenecks,
                       ground_truth_input: train_ground_truth})
        tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
                        (datetime.now(), i, train_accuracy * 100))
        tf.logging.info('%s: Step %d: Cross entropy = %f' %
                        (datetime.now(), i, cross_entropy_value))
        validation_bottlenecks, validation_ground_truth, _ = (
            get_random_cached_bottlenecks(
                sess, image_lists, FLAGS.validation_batch_size, 'validation',
                FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                FLAGS.architecture))
        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        validation_summary, validation_accuracy = sess.run(
            [merged, evaluation_step],
            feed_dict={bottleneck_input: validation_bottlenecks,
                       ground_truth_input: validation_ground_truth})
        validation_writer.add_summary(validation_summary, i)
        tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                        (datetime.now(), i, validation_accuracy * 100,
                         len(validation_bottlenecks)))

      # Store intermediate results
      intermediate_frequency = FLAGS.intermediate_store_frequency

      if (intermediate_frequency > 0 and (i % intermediate_frequency == 0)
          and i > 0):
        intermediate_file_name = (FLAGS.intermediate_output_graphs_dir +
                                  'intermediate_' + str(i) + '.pb')
        tf.logging.info('Save intermediate result to : ' +
                        intermediate_file_name)
        save_graph_to_file(sess, graph, intermediate_file_name)

    # We've completed all our training, so run a final test evaluation on
    # some new images we haven't used before.
    test_bottlenecks, test_ground_truth, test_filenames = (
        get_random_cached_bottlenecks(
            sess, image_lists, FLAGS.test_batch_size, 'testing',
            FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
            decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
            FLAGS.architecture))
    test_accuracy, predictions = sess.run(
        [evaluation_step, prediction],
        feed_dict={bottleneck_input: test_bottlenecks,
                   ground_truth_input: test_ground_truth})
    tf.logging.info('Final test accuracy = %.1f%% (N=%d)' %
                    (test_accuracy * 100, len(test_bottlenecks)))

    if FLAGS.print_misclassified_test_images:
      tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
      for i, test_filename in enumerate(test_filenames):
        if predictions[i] != test_ground_truth[i].argmax():
          tf.logging.info('%70s  %s' %
                          (test_filename,
                           list(image_lists.keys())[predictions[i]]))

    # Write out the trained graph and labels with the weights stored as
    # constants.
    save_graph_to_file(sess, graph, FLAGS.output_graph)
    with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
      f.write('\n'.join(image_lists.keys()) + '\n')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_dir',
      type=str,
      default='',
      help='Path to folders of labeled images.'
  )
  parser.add_argument(
      '--output_graph',
      type=str,
      default='/tmp/output_graph.pb',
      help='Where to save the trained graph.'
  )
  parser.add_argument(
      '--intermediate_output_graphs_dir',
      type=str,
      default='/tmp/intermediate_graph/',
      help='Where to save the intermediate graphs.'
  )
  parser.add_argument(
      '--intermediate_store_frequency',
      type=int,
      default=0,
      help="""\
         How many steps to store intermediate graph. If "0" then will not
         store.\
      """
  )
  parser.add_argument(
      '--output_labels',
      type=str,
      default='/tmp/output_labels.txt',
      help='Where to save the trained graph\'s labels.'
  )
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.'
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=4000,
      help='How many training steps to run before ending.'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='How large a learning rate to use when training.'
  )
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a test set.'
  )
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a validation set.'
  )
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=10,
      help='How often to evaluate the training results.'
  )
  parser.add_argument(
      '--train_batch_size',
      type=int,
      default=100,
      help='How many images to train on at a time.'
  )
  parser.add_argument(
      '--test_batch_size',
      type=int,
      default=-1,
      help="""\
      Berapa banyak gambar untuk diuji. Set tes ini hanya digunakan satu kali, untuk mengevaluasi
      akurasi akhir model setelah pelatihan selesai.
      Nilai -1 menyebabkan seluruh set tes digunakan, yang mengarah ke lebih banyak
      hasil yang stabil di seluruh proses.\
      """
  )
  parser.add_argument(
      '--validation_batch_size',
      type=int,
      default=100,
      help="""\
      Berapa banyak gambar untuk digunakan dalam kumpulan evaluasi. Set validasi ini adalah
      digunakan jauh lebih sering daripada set tes, dan merupakan indikator awal bagaimana
      model yang akurat adalah selama pelatihan.
      Nilai -1 menyebabkan seluruh set validasi digunakan, yang mengarah ke
      hasil yang lebih stabil di seluruh iterasi pelatihan, tetapi mungkin lebih lambat pada yang besar
      set pelatihan.\
      """
  )
  parser.add_argument(
      '--print_misclassified_test_images',
      default=False,
      help="""\
      Whether to print out a list of all misclassified test images.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path untuk classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--bottleneck_dir',
      type=str,
      default='/tmp/bottleneck',
      help='Path to cache bottleneck layer values as files.'
  )
  parser.add_argument(
      '--final_tensor_name',
      type=str,
      default='final_result',
      help="""\
      The name of the output classification layer in the retrained graph.\
      """
  )
  parser.add_argument(
      '--flip_left_right',
      default=False,
      help="""\
      Whether to randomly flip half of the training images horizontally.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--random_crop',
      type=int,
      default=0,
      help="""\
      A percentage determining how much of a margin to randomly crop off the
      training images.\
      """
  )
  parser.add_argument(
      '--random_scale',
      type=int,
      default=0,
      help="""\
      A percentage determining how much to randomly scale up the size of the
      training images by.\
      """
  )
  parser.add_argument(
      '--random_brightness',
      type=int,
      default=0,
      help="""\
      A percentage determining how much to randomly multiply the training image
      input pixels up or down by.\
      """
  )
  parser.add_argument(
      '--architecture',
      type=str,
      default='inception_v3',
      help="""\
      Model arsitektur mana yang digunakan. 'inception_v3' adalah yang paling akurat, tetapi
      juga yang paling lambat. Untuk model yang lebih cepat atau lebih kecil, pilih MobileNet dengan
      bentuk 'mobilenet_ <ukuran parameter> _ <input_size> [_quantized]'. Sebagai contoh,
      'mobilenet_1.0_224' akan memilih model yang berukuran 17 MB dan membutuhkan 224
      gambar input piksel, sementara 'mobilenet_0.25_128_quantized' akan memilih banyak yang
      kurang akurat, tetapi jaringan lebih kecil dan lebih cepat yaitu 920 KB pada disk dan
      membutuhkan 128x128 gambar. Lihat https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html
      untuk informasi lebih lanjut tentang Mobilenet.\
      """)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
