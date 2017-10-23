clear all;
clc
close all;
test=[0.2 0.3 0.4 0.5 0.6 0.7];
val_mnb = [0.7446 0.7420 0.7391 0.7427 0.7410 0.7409];
train_mnb = [0.7402 0.7412 0.7413 0.7394 0.7411 0.7402];
val_lr = [0.8810 0.8794 0.8780 0.8791 0.8779 0.8774];
train_lr = [0.8789 0.8791 0.8797 0.8779 0.8798 0.8783];
val_svm = [0.8805 0.8785 0.8778 0.8789 0.8781 0.8779];
train_svm = [0.8783 0.8788 0.8790 0.8775 0.8802 0.8789];
val_lda = [0.8645 0.8622 0.8618 0.8636 0.8630 0.8617];
train_lda = [0.8623 0.8630 0.8632 0.8622 0.8637 0.8632];
val_rf = [0.9158 0.9130 0.9108 0.9102 0.9059 0.9044];
train_rf = [0.9922 0.9923 0.9922 0.9923 0.9926 0.9930];
figure()
hold on
size = 14
plot(test,train_mnb,'r-o','DisplayName','MNB_ train','MarkerSize',size);
plot(test,train_lr,'g-o','DisplayName','LR_ train','MarkerSize',size);
plot(test,train_svm,'b-o','DisplayName','SVM_ train','MarkerSize',size);
plot(test,train_lda,'m-o','DisplayName','LDA_ train','MarkerSize',size);
plot(test,train_rf,'-o','Color',[0. 0.9 0.89],'DisplayName','RF_ train','MarkerSize',size);

plot(test,val_mnb,'r-*','DisplayName','MNB_ val','MarkerSize',size);
plot(test,val_lr,'g-*','DisplayName','LR_ val','MarkerSize',size);
plot(test,val_svm,'b-*','DisplayName','SVM_ val','MarkerSize',size);
plot(test,val_lda,'m-*','DisplayName','LDA_ train','MarkerSize',size);
plot(test,val_rf,'-*','Color',[0. 0.9 0.89],'DisplayName','RF_ train','MarkerSize',size);

legend('show')
labels = {'MNB_ train','LR_ train','SVM_ train','LDA_ train','RF_ train','MNB','LR','SVM','LDA','RF'};

lgd = legend('Location','northeast');
lgd.FontSize = 20;
xlabel('size of test data')
ylabel('accuracy')
title('Accuracy by different models on different sizes of test data')
xticks([0.2 0.3 0.4 0.5 0.6 0.7])
set(gca,'fontsize',20)