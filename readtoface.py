import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random
import numpy as np
from efficientnet_pytorch import EfficientNet

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


#seed 설정
seed = 719
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
#하이퍼 파라미터 설정
batch_size = 64  # 학습 배치 크기
test_batch_size = 1000  # 테스트 배치 크기 (학습 과정을 제외하므로 더 큰 배치 사용 가능)
max_epochs = 10  # 학습 데이터셋 총 훈련 횟수
lr = 0.01  # 학습률
momentum = 0.5  # SGD에 사용할 모멘텀 설정 (파라미터 업데이트 시 관성 효과 사용)
log_interval = 200  # interval 때마다 로그 남김
use_cuda = torch.cuda.is_available()  # GPU cuda 사용 여부 확인
device = torch.device("cuda" if use_cuda else "cpu")  # GPU cuda 사용하거나 없다면 CPU 사용
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}  # num_workers: data loading할 프로세스 수, pin_memory: 고정된 메모리 영역 사용
#모델 및 활성화 함수 세팅
model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=15).to(device)
base_optimizer = optim.SGD
optimizer = SAM(model.parameters(), base_optimizer, lr=lr, momentum=momentum)  # 최적화 알고리즘 정의 (SGD와 SAM사용)
criterion = nn.CrossEntropyLoss()  # 손실 함수 정의 (CrossEntropy 사용)

def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()  # 모델 학습 모드 설정
    summary_loss = AverageMeter()  # 학습 손실값 기록 초기화
    summary_acc = AverageMeter() # 학습 정확도 기록 초기화
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # 현재 미니 배치의 데이터, 정답 불러옴
        optimizer.zero_grad()  # gradient 0으로 초기화
        output = model(data)  # 모델에 입력값 feed-forward
        loss = criterion(output, target)  # 예측값(클래스 별 score)과 정답간의 손실값 계산
        loss.backward()  # 손실값 역전파 (각 계층에서 gradient 계산, pytorch는 autograd로 gradient 자동 계산)
        
        # SAM 내용 추가
        optimizer.first_step(zero_grad=True)  # 모델의 파라미터 업데이트 (gradient 이용하여 파라미터 업데이트)
        criterion(model(data), target).backward()
        optimizer.second_step(zero_grad=True)


        summary_loss.update(loss.detach().item())  # 손실값 기록
        pred = output.argmax(dim=1, keepdim=True)  # 예측값 중에서 최고 score를 달성한 클래스 선발
        correct = pred.eq(target.view_as(pred)).sum().item()  # 정답과 예측 클래스가 일치한 개수
        summary_acc.update(correct / data.size(0))  # 정확도 기록
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}, Accuracy: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), summary_loss.avg, summary_acc.avg))
            
    return summary_loss.avg, summary_acc.avg

def test(log_interval, model, device, test_loader):
    model.eval()  # 모델 검증 모드 설정 (inference mode)
    summary_loss = AverageMeter()  # 테스트 손실값 기록 초기화
    summary_acc = AverageMeter() # 테스트 정확도 기록 초기화
    with torch.no_grad():  # 검증 모드이므로 gradient 계산안함
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # 현재 미니 배치의 데이터, 정답 불러옴
            output = model(data)  # 모델에 입력값 feed-forward
            loss = criterion(output, target)  # 예측값(클래스 별 score)과 정답간의 손실값 계산
            summary_loss.update(loss.detach().item())  # 손실값 기록
            pred = output.argmax(dim=1, keepdim=True)  # 예측값 중에서 최고 score를 달성한 클래스 선발
            correct = pred.eq(target.view_as(pred)).sum().item()  # 정답과 예측 클래스가 일치한 개수
            summary_acc.update(correct / data.size(0))  # 정확도 기록

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.6f}\n'.format
          (summary_loss.avg, summary_acc.avg))  # 정답을 맞춘 개수 / 테스트셋 샘플 수 -> Accuracy

    return summary_loss.avg, summary_acc.avg

best_acc = 0
best_epoch = 0
for epoch in range(1, max_epochs+1):
    train_loss, train_acc = train(log_interval, model, device, train_loader, optimizer, epoch)
    test_loss, test_acc = test(log_interval, model, device, test_loader)

    # 테스트에서 best accuracy 달성하면 모델 저장
    if test_acc > best_acc:
        best_acc = test_acc
        best_epoch = epoch
        # torch.save(model, os.path.join(workspace_path, f'cifar10_cnn_model_best_acc_{best_epoch}-epoch.pt'))
        print(f'# save model: cifar10_cnn_model_best_acc_{best_epoch}-epoch.pt\n')

print(f'\n\n# Best accuracy model({best_acc * 100:.2f}%): cifar10_cnn_model_best_acc_{best_epoch}-epoch.pt\n')