기본적인 작동방식:
from EMCCD_library import EMCCD로 불러오면 되고
cam = EMCCD()하면 됨.

기본값은 single_scan(Kinetic series)

EMCCD 측에서 쓰레드를 두 개 가지고 있음.
_ImageGetThread, _TemperatureThread(QThread 상속)
이 둘 다 동시에 여러 개 가지고 있어야 할 쓰레드는 아니기 때문에
EMCCD가 각각을 하나만 가지도록 하고 외부에서 요구하면 가져갈 수 있음
(getter setter는 아니고 별도로 get_temperature_thread(), get_image_getting_thread()로 해 놓았음.)

이 중에서, 쿨러는 어차피 온도 감시 쓰레드를 항상 켜 놓아야 해서 사실상 외부에서 필요로 하지는 않을 것임. cooler_on과 cooler_off로 쿨러를 켜고 끌 수는 있음.
이미지의 경우, 외부에서 _ImageGetThread를 받아갔다면, 이제 start를 해주면 됨
(UI python script에서 StartAcquisition()에 들어갈 코드)

EMCCD에서 현재는 시그널을 두개 가지고 있음
하나는 new_image_signal = pyqtSignal(np.ndarray)
다른 하나는 temperature_signal = pyqtSignal(int, str)

temperature_signal은 3초마다 현재 센서의 온도와 driver code를 함께 가지고 emit하도록 되어 있음
new_image_signal은 startAcquisition 이후 이미지를 새로 받을 때 마다 해당 이미지를 가지고 emit 하도록 되어 있음.



property(getter)

acquisition_mode : "single_scan" or "continuous_scan". getter
ccd_image : image(np.ndarray). getter
current_temperature. getter
exposure_time : # 정확한 exposuretime은 startAcquisition 이후 결정됨. gette
gain. getter
sensor_size : [raw_image_width, raw_image_height]. getter
status. getter
target_temperature. getter
trigger_count : single scan(kinetic series일 때 유효). getter


property.setter

acquisition_mode
exposure_time
gain
target_temperature
trigger_count


function

cooler_on
cooler_off
get_image_acquiring_thread


앞으로 추가할 것
SetImage에서 Crop mode 추가하기(항상 전체 이미지를 찍는건 비효율)
Binning도 쓸모 있을 수 있음
external trigger 설정
gain 300위로 늘리려면 advanced gain 활성화해야 하는데 그거 설정 추가
Acquisition 중 exposure time 변경 가능하게 바꾸기(아마 바로는 안되고 임의로 중지했다가 다시 재개해야 할듯)
