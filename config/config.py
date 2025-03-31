import os
from detectron2.config import get_cfg
from detectron2 import model_zoo

def get_medication_classes():
    """
    약물 분류 클래스 목록 반환
    """
    return [
        '보령부스파정 5mg', '뮤테란캡슐 100mg', '일양하이트린정 2mg', '기넥신에프정(은행엽엑스)(수출용)', 
        '무코스타정(레바미피드)(비매품)', '알드린정', '뉴로메드정(옥시라세탐)', '타이레놀정500mg', 
        '에어탈정(아세클로페낙)', '삼남건조수산화알루미늄겔정', '타이레놀이알서방정(아세트아미노펜)(수출용)', 
        '삐콤씨에프정 618.6mg/병', '조인스정 200mg', '쎄로켈정 100mg', '리렉스펜정 300mg/PTP', 
        '아빌리파이정 10mg', '자이프렉사정 2.5mg', '다보타민큐정 10mg/병', '써스펜8시간이알서방정 650mg', 
        '에빅사정(메만틴염산염)(비매품)', '리피토정 20mg', '크레스토정 20mg', '가바토파정 100mg', 
        '동아가바펜틴정 800mg', '오마코연질캡슐(오메가-3-산에틸에스테르90)', '란스톤엘에프디티정 30mg', 
        '리리카캡슐 150mg', '종근당글리아티린연질캡슐(콜린알포세레이트)\xa0', '콜리네이트연질캡슐 400mg', 
        '트루비타정 60mg/병', '스토가정 10mg', '노바스크정 5mg', '마도파정', '플라빅스정 75mg', 
        '엑스포지정 5/160mg', '펠루비정(펠루비프로펜)', '아토르바정 10mg', '라비에트정 20mg', 
        '리피로우정 20mg', '자누비아정 50mg', '맥시부펜이알정 300mg', '메가파워정 90mg/병', 
        '쿠에타핀정 25mg', '비타비백정 100mg/병', '놀텍정 10mg', '자누메트정 50/850mg', 
        '큐시드정 31.5mg/PTP', '아모잘탄정 5/100mg', '세비카정 10/40mg', '트윈스타정 40/5mg', 
        '카나브정 60mg', '울트라셋이알서방정', '졸로푸트정 100mg', '트라젠타정(리나글립틴)', 
        '비모보정 500/20mg', '레일라정', '리바로정 4mg', '렉사프로정 15mg', '트라젠타듀오정 2.5/850mg', 
        '낙소졸정 500/20mg', '아질렉트정(라사길린메실산염)', '자누메트엑스알서방정 100/1000mg', 
        '글리아타민연질캡슐', '신바로정', '에스원엠프정 20mg', '브린텔릭스정 20mg', 
        '글리틴정(콜린알포세레이트)', '제미메트서방정 50/1000mg', '아토젯정 10/40mg', '로수젯정10/5밀리그램', 
        '로수바미브정 10/20mg', '카발린캡슐 25mg', '케이캡정 50mg'
    ]

def setup_cfg(data_root, image_data_root, output_path, resume_from_checkpoint=False, checkpoint_path=None):
    """
    모델 설정 생성
    
    Args:
        data_root: 데이터 루트 경로
        image_data_root: 이미지 데이터 경로
        output_path: 출력 경로
        resume_from_checkpoint: 체크포인트에서 이어서 학습할지 여부
        checkpoint_path: 체크포인트 경로 (resume_from_checkpoint가 True인 경우에만 사용)
        
    Returns:
        cfg: 설정 객체
    """
    # 설정 객체 생성
    cfg = get_cfg()
    
    # Cascade R-CNN 기본 config 불러오기
    cfg.merge_from_file("configs/Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml")
    
    # 데이터셋 이름 지정
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    
    # 클래스 수 (배경 포함 X)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(get_medication_classes())
    
    # 디바이스 설정
    cfg.MODEL.DEVICE = "cuda"
    cfg.DATALOADER.NUM_WORKERS = 2
    
    # Batch size 설정 
    cfg.SOLVER.IMS_PER_BATCH = 8
    
    # Learning Rate
    cfg.SOLVER.BASE_LR = 0.0001
    
    # Iteration 수
    cfg.SOLVER.MAX_ITER = 20000
    
    # 웨이트 감소 설정
    cfg.SOLVER.WEIGHT_DECAY = 0.0005
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0  # 정규화 레이어에는 웨이트 감소 미적용
    
    # ROI 헤드에 드롭아웃 추가
    cfg.MODEL.ROI_BOX_HEAD.DROPOUT_PROBABILITY = 0.2
    
    # 원본 이미지 크기를 고려한 다양한 크기 설정
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.TYPE = "relative_range"
    cfg.INPUT.CROP.SIZE = [0.9, 0.9]
    
    # 색상 변형 추가
    cfg.INPUT.COLOR_AUG_SSD = True
    
    # 마스크 분할 비활성화
    cfg.MODEL.MASK_ON = False
    
    # 포컬 로스 파라미터
    cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA = 2.0
    cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA = 0.25
    
    # Cascade R-CNN의 핵심 파라미터
    cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS = (0.5, 0.6, 0.7)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
    
    # 객체 제안 품질에 영향
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 3000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1500
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1500
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 750
    
    # 앵커 설정 조정
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    
    # 출력 폴더 설정
    cfg.OUTPUT_DIR = output_path
    os.makedirs(output_path, exist_ok=True)
    
    # 체크포인트에서 이어서 학습할 경우
    if resume_from_checkpoint and checkpoint_path is not None:
        cfg.MODEL.WEIGHTS = checkpoint_path
    else:
        # 처음부터 학습할 경우 (pretrained weight 사용)
        cfg.MODEL.WEIGHTS = os.path.join(data_root, "model_final_480dd8.pkl")
    
    return cfg

def setup_test_cfg(output_path, classes, weight_path=None):
    """
    테스트를 위한 설정 생성
    
    Args:
        output_path: 출력 경로
        classes: 클래스 목록
        weight_path: 사용할 모델 가중치 경로
        
    Returns:
        cfg: 테스트 설정 객체
    """
    cfg = get_cfg()
    
    # Cascade Mask R-CNN config 불러오기
    cfg.merge_from_file("configs/Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml")
    
    # 클래스 수 설정
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
    
    # Mask 사용 안 함
    cfg.MODEL.MASK_ON = False
    
    # 학습한 weight 불러오기
    if weight_path:
        cfg.MODEL.WEIGHTS = weight_path
    else:
        cfg.MODEL.WEIGHTS = os.path.join(output_path, "model_final.pth")
    
    # GPU 사용
    cfg.MODEL.DEVICE = "cuda"
    
    # 테스트 시 입력 이미지 크기 설정 (학습과 동일하게)
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1260

    # Cascade R-CNN의 핵심 파라미터
    cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS = (0.5, 0.6, 0.7)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4

    # 앵커 설정 조정
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    
    return cfg