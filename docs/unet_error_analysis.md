# Phân tích điều kiện dự đoán đúng và sai của hai mô hình U-Net

Trên các bảng kết quả test đã lưu, U-Net từ đầu đạt mean per-image Dice
0.7636, còn U-Net ResNet34 pretrained đạt 0.8968. Cả hai đạt điểm cao nhất
ở nhóm polyp chiếm 5–<20% diện tích ảnh; các nhóm chiếm 1–<5% hoặc chạm
mép ảnh có điểm thấp hơn. ResNet34 cải thiện kết quả trung bình nhưng vẫn
có trường hợp bỏ sót vùng mask hoặc dự đoán thừa, và không tốt hơn ở mọi ảnh.

Phân tích dùng [số liệu gộp trong experiments.json](../report/results/experiments.json):
các khóa `test_by_size`, `test_by_border`, `test_error_frequency` và
`test_per_image_preview` của mỗi run, cùng hình tốt/xấu. Những nhận xét về màu sắc, hình dạng và đường
biên được rút từ ảnh minh họa; phần giải thích nguyên nhân là giả thuyết
phù hợp với các biểu hiện quan sát được, chưa phải kết quả thí nghiệm cô lập
nguyên nhân. Phạm vi là các run đã lưu ở một seed; việc đối chiếu đầy đủ
ID train/validation/test gốc của hai run vẫn theo
[hướng dẫn kiểm tra split](unet_experiments.md#3-chạy-lại-bảng-so-sánh).

## 1. Những điều kiện đã có thống kê trên toàn bộ test

Diện tích dưới đây là số pixel mask thật chia cho số pixel ảnh 256×256,
không phải kích thước thực của polyp. Chạm mép nghĩa là mask thật có pixel
ở hàng hoặc cột ngoài cùng của ảnh.

| Điều kiện | Số ảnh mỗi run | Mean Dice U-Net | Mean Dice ResNet34 | Nhận xét từ số liệu |
|---|---:|---:|---:|---|
| Diện tích 1–<5% | 19 | 0.6052 | 0.8705 | U-Net từ đầu kém nhất trong ba nhóm diện tích; ResNet34 cải thiện nhưng nhóm này vẫn thấp hơn nhóm 5–<20% |
| Diện tích 5–<20% | 47 | 0.8323 | 0.9168 | Nhóm có mean Dice cao nhất ở cả hai mô hình |
| Diện tích >=20% | 34 | 0.7571 | 0.8838 | Thấp hơn nhóm 5–<20%; cần chú ý vùng mask chưa được phủ hết |
| Không chạm mép ảnh | 79 | 0.7773 | 0.9095 | Cao hơn nhóm chạm mép trong mỗi run |
| Chạm mép ảnh | 21 | 0.7121 | 0.8488 | Cả hai khó phủ đủ mask hơn, thể hiện ở Recall thấp hơn |

Ở nhóm diện tích 1–<5%, U-Net có mean Precision 0.6178 và Recall 0.7286,
phù hợp với tình trạng dự đoán thừa đáng kể ở một số ảnh. Không thể chỉ từ
điểm Dice thấp mà kết luận mọi ảnh trong nhóm đều bị bỏ sót. ResNet34 đạt
Precision 0.8690 và Recall 0.8976, cho thấy cả độ chính xác của vùng dự đoán
và mức bao phủ vùng thật đều tốt hơn trong bảng nhóm đã lưu.

Ở nhóm diện tích >=20%, mean Precision/Recall lần lượt là 0.9028/0.6955
với U-Net và 0.9514/0.8436 với ResNet34. Precision cao hơn Recall cùng các
hình minh họa phù hợp với lỗi chỉ giữ phần rõ nhất của vùng polyp và bỏ sót
phần còn lại. Đây là biểu hiện dự đoán thiếu, không tự xác định nguyên nhân.

Với ảnh chạm mép, Recall giảm từ 0.8351 xuống 0.6487 ở U-Net và từ 0.9255
xuống 0.7984 ở ResNet34 so với nhóm không chạm mép. Một giải thích có thể
là phần tổn thương sát mép không có đầy đủ hình dạng và ngữ cảnh xung quanh.
Các nhóm có thể khác nhau cả về diện tích lẫn hình dạng, nên chênh lệch này
chưa chứng minh riêng việc chạm mép gây ra lỗi.

Nguồn: các khóa `test_by_size` và `test_by_border` của `unet_kvasir` và
`unet_resnet34_kvasir` trong [experiments.json](../report/results/experiments.json).

## 2. U-Net từ đầu thường làm tốt và gặp khó ở đâu

### Các trường hợp làm tốt

Trong [năm ảnh Dice cao nhất](../report/results/unet_kvasir/test_best_cases.png),
vùng mask nhìn tương đối tập trung, có hình dạng gọn và ranh giới với vùng
xung quanh dễ theo dõi bằng mắt. Vùng xanh lá chiếm phần lớn mask, còn FP
và FN chủ yếu là những dải nhỏ gần đường biên. Dice của năm ảnh này nằm
trong khoảng 0.961–0.981 theo số hiển thị trên hình.

Các ảnh này minh họa việc mô hình có thể học vị trí và đường biên khi đặc
trưng vùng tổn thương đủ rõ. Điều kiện có bằng chứng tần suất ở mức nhóm
là diện tích 5–<20% và không chạm mép; mô tả “ranh giới dễ theo dõi” mới là
quan sát định tính của các ảnh được chọn, chưa phải nhãn đã chấm cho 100 ảnh.

### Chọn nhầm cấu trúc có vẻ ngoài tương tự

Ở hàng đầu của [năm ảnh Dice thấp nhất](../report/results/unet_kvasir/test_worst_cases.png),
mask thật nằm ở phần phía trên ảnh nhưng mô hình dự đoán một vùng khác ở
phía dưới. Ảnh có nhiều cấu trúc mô nhô lên với màu và vùng sáng tương tự;
overlay thể hiện vùng thật bị tô xanh dương và vùng sai bị tô đỏ. Dice bằng 0
vì hai vùng không giao nhau, không phải vì mô hình không tạo mask.

Một giả thuyết là mô hình dựa nhiều vào dấu hiệu cục bộ như vùng sáng,
kết cấu hoặc hình khối nhô lên mà chưa phân biệt tốt cấu trúc mục tiêu với
vùng nền có vẻ ngoài gần giống. Hai ảnh có TP bằng 0 trong thống kê toàn test
cho thấy lỗi chọn sai vị trí có xảy ra, nhưng không đủ để quy mọi trường hợp
TP bằng 0 cho cùng một đặc điểm ảnh.

### Bỏ sót vùng hẹp và dự đoán thừa quanh vùng nhỏ

Hàng thứ tư của hình ảnh lỗi có mask thật hẹp, kéo dài; dự đoán chỉ giữ một
phần nhỏ, với Dice 0.129. Hàng thứ ba và thứ năm có vùng mask thật nhỏ hơn
rõ rệt vùng dự đoán hoặc có nhiều mảnh FP tách rời. Đây là hai dạng lỗi khác
nhau: vùng mảnh bị bỏ sót, còn vùng nhỏ có thể bị mở rộng hoặc bị lẫn với nền.

Khi resize ảnh về 256×256, các phần hẹp có thể còn rất ít pixel. Encoder
lại giảm độ phân giải qua pooling, nên một giả thuyết là các đặc trưng nhỏ
khó được giữ và khôi phục đầy đủ. Về mặt metric, với mask nhỏ, cùng một số
pixel FP/FN sẽ chiếm tỷ lệ lớn hơn trong mẫu số Dice. Hai yếu tố này có thể
góp phần làm nhóm 1–<5% khó hơn; hình và bảng hiện tại chưa tách được ảnh
hưởng của resize, pooling, dữ liệu huấn luyện và hình dạng.

### Khả năng tổng quát hóa còn hạn chế

Trong `runs.unet_kvasir.training_history` của [số liệu đã lưu](../report/results/experiments.json),
Train Dice tăng từ 0.8256 ở epoch 32 lên 0.9484 ở epoch 42, trong khi
Validation Dice không vượt 0.7634; validation loss tăng từ 0.2748 lên 0.3230.
Đây là dấu hiệu overfitting ở giai đoạn cuối. Việc học tốt thêm trên train
không đồng nghĩa xử lý tốt hơn các hình dạng và nền khác trên dữ liệu giữ lại.
Kết quả test sử dụng checkpoint epoch 32, không phải epoch 42.

## 3. U-Net ResNet34 pretrained làm tốt hơn ở đâu và còn sai gì

### Phủ vùng và bám biên tốt hơn trong nhiều trường hợp

Trong [năm ảnh tốt nhất](../report/results/unet_resnet34_kvasir/test_best_cases.png),
các vùng TP phủ gần hết mask thật và FP/FN còn ít quanh biên. Những ví dụ
này gồm cả vùng tập trung ở giữa ảnh và vùng bị cắt ở mép. Do đó “chạm mép”
là điều kiện có điểm trung bình thấp hơn, không phải điều kiện chắc chắn sai.

Trên toàn bộ test, 84 ảnh có Dice >=0.8 so với 59 ảnh của U-Net từ đầu;
không có ảnh Dice <0.5 so với 10 ảnh ở baseline. Nhóm 1–<5% có mean Dice
0.8705, còn nhóm 5–<20% đạt 0.9168. Các số này thể hiện mức cải thiện về
phân đoạn trong các run đã lưu, không bảo đảm mô hình luôn tốt hơn ở từng ảnh.

Encoder pretrained đã có điểm khởi đầu học từ ImageNet, nên việc chuyển
giao đặc trưng là một giả thuyết giải thích cho kết quả tốt hơn. Thí nghiệm
cũng thay encoder, decoder và chuẩn hóa đầu vào, nên chưa thể tách riêng
đóng góp của pretrained weights. Muốn kiểm tra riêng giả thuyết này cần
so sánh cùng kiến trúc ResNet34 với khởi tạo ngẫu nhiên và pretrained.

### Vẫn bỏ sót một phần vùng mask có hình dạng phức tạp

Ở hàng đầu của [năm ảnh lỗi lớn nhất](../report/results/unet_resnet34_kvasir/test_worst_cases.png),
mask thật kéo dài qua nhiều phần nối nhau nhưng dự đoán chỉ giữ hai phần
tách rời, bỏ phần giữa; Dice 0.613. Hàng thứ tư cũng bỏ sót một phần lớn của
mask kéo lên mép phải, dù phần tập trung phía dưới được nhận diện.

Các ví dụ gợi ý mô hình có thể giữ các phần có đặc trưng nổi bật nhưng
không phủ đủ hình dạng toàn bộ vùng gán nhãn. ResNet34 vẫn giảm độ phân
giải xuống 8×8 với ảnh 256×256, nên pretrained không tự bảo đảm giữ được
mọi cấu trúc mảnh. Giải thích này là giả thuyết; chưa có thống kê riêng theo
độ mảnh, số vùng mask hoặc độ phức tạp đường biên.

### Vẫn dự đoán thừa sang vùng mô liền kề

Hàng thứ hai có cả phần polyp thật được phủ và một vùng FP lớn phía dưới.
Hàng thứ ba có dự đoán kéo dài lên trên vùng mask thật; vùng mô mục tiêu và
vùng liền kề có màu gần nhau trong ảnh. Sự chuyển tiếp màu/độ sáng không rõ
có thể khiến ranh giới dự đoán lan sang nền. Trên toàn test, vẫn có một ảnh
Precision <0.5 và hai ảnh Recall <0.5, nên kết quả trung bình cao không loại
bỏ lỗi dự đoán thừa hoặc thiếu ở các trường hợp khó.

### Vẫn còn khoảng cách train và validation

Ở epoch 30, Train Dice là 0.9847 còn Validation Dice là 0.8834. Mô hình vẫn
có khoảng cách tổng quát hóa; checkpoint tốt nhất được chọn ở epoch 20 với
Validation Dice 0.8874. Không dùng điểm train gần 1 để kết luận mô hình đúng
gần như tuyệt đối trên ảnh mới.

## 4. Ba ví dụ đối chiếu có số liệu từng ảnh

Bảng dùng các hàng thực sự xuất hiện trong `test_per_image_preview` của
hai run trong [experiments.json](../report/results/experiments.json).
Đây là các ví dụ xác định, không phải phép ghép đủ 100 ảnh.

| ID ảnh | Dice U-Net | Dice ResNet34 | Điều quan sát được |
|---|---:|---:|---|
| `cju3ya7goj6at0818v2l5ay7f` | 0.4789 | 0.6548 | Mask có hai vùng tách nhau; ResNet34 giảm FP nhưng vẫn bỏ sót một vùng trong hình lỗi |
| `cju8abobpqbir08189u01huru` | 0.9081 | 0.9889 | Mask chạm mép vẫn được ResNet34 phủ gần hết; ảnh nằm ở hàng đầu hình tốt nhất của ResNet34 |
| `cju1brhsj3rls0855a1vgdlen` | 0.6990 | 0.6343 | ResNet34 mở rộng mask sang vùng liền kề và đạt Dice thấp hơn U-Net |

Ở ảnh thứ nhất, FP giảm từ 2.128 xuống 668 pixel trong khi FN chỉ giảm
từ 847 xuống 811. Cải thiện Dice chủ yếu đi kèm giảm dự đoán thừa; Recall
vẫn chỉ tăng từ 0.6174 lên 0.6337. Không nên mô tả ảnh này là đã tìm đủ
mọi vùng polyp.

Ở ảnh thứ hai, FP giảm từ 337 xuống 104 và FN giảm từ 908 xuống 53,
cho thấy cả ranh giới và độ bao phủ đều cải thiện trên mask đã resize.

Ở ảnh thứ ba, FP tăng từ 2.253 lên 2.786 và FN tăng từ 2 lên 109;
Precision giảm từ 0.5375 xuống 0.4740. Đây là ví dụ trực tiếp rằng mô hình
pretrained có thể kém hơn trên một ảnh dù tốt hơn về trung bình.

## 5. Kết luận theo điều kiện ảnh

**U-Net từ đầu:** trong các nhóm đã thống kê, mô hình đạt kết quả cao nhất
với vùng polyp chiếm 5–<20% diện tích ảnh và cao hơn ở nhóm không chạm mép.
Các ảnh Dice cao minh họa vùng tổn thương có hình dạng tập trung và ranh giới
dễ theo dõi. Các ảnh lỗi minh họa việc chọn nhầm cấu trúc mô gần giống,
bỏ sót vùng hẹp/kéo dài và mở rộng dự đoán quanh vùng nhỏ. Có 59 ảnh Dice
>=0.8, 10 ảnh Dice <0.5 và 12 ảnh bỏ sót hơn một nửa số pixel polyp.

**U-Net ResNet34 pretrained:** các bảng cho thấy kết quả tốt hơn ở từng
nhóm diện tích và từng nhóm chạm mép/không chạm mép đã thống kê. Mô hình
phủ vùng và bám biên tốt trong nhiều ảnh; 84 ảnh có Dice >=0.8 và không có
ảnh Dice <0.5 trong test này. Điểm thấp hơn tương đối vẫn xuất hiện ở nhóm
1–<5% và nhóm chạm mép; ảnh minh họa cho thấy mô hình còn bỏ sót một phần
mask phức tạp/nhiều vùng hoặc lan sang mô liền kề có vẻ ngoài tương tự.

Như vậy, kết luận “thường tốt hơn/kém hơn trong điều kiện nào” được hỗ trợ
định lượng cho **các nhóm diện tích và chạm mép**. Nhận xét về ranh giới rõ,
màu tương tự, vùng mảnh và nhiều vùng chỉ mô tả các ví dụ đã xem. Cần gán
nhãn các đặc điểm đó cho toàn bộ test để thống kê tần suất theo điều kiện.
Tập này không có nhóm dưới 1% diện tích; độ mờ và phản sáng cũng chưa được
định lượng, nên chưa mở rộng kết luận sang các điều kiện đó.

## 6. Giới hạn của giải thích nguyên nhân

Một vùng màu đỏ trên overlay xác nhận FP, vùng xanh dương xác nhận FN;
chúng không tự cho biết vì sao mạng tạo ra lỗi. Các giả thuyết về đặc trưng
cục bộ, mất chi tiết sau resize/giảm độ phân giải và thiếu ngữ cảnh ở mép
là các cách giải thích cần kiểm tra thêm. Có thể phân tích các giả thuyết
trên validation bằng nhóm đặc điểm ảnh hoặc ablation tương ứng; giữ test
hiện tại để báo cáo kết quả cuối, không chọn tham số theo các ảnh test khó.

Các ảnh tốt/xấu được chọn theo Dice cao nhất/thấp nhất nên hữu ích để minh
họa, không đại diện cho phân bố toàn bộ tập. Các nhóm diện tích và chạm mép
có thể chồng lấn về đặc điểm. Bảng nguồn được in làm tròn và còn thiếu CSV
từng ảnh đầy đủ, nên chưa đếm số ảnh ResNet34 cải thiện/giảm chất lượng trên
toàn bộ 100 ảnh. Điều này không ngăn việc phân tích các số tổng hợp và các
ví dụ cụ thể đã có, nhưng giới hạn sức mạnh của kết luận nguyên nhân.
