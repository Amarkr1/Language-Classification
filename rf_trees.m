close all;
clc
n = [100 200 300 400 500 600];
val = [0.908264622209 0.909313370944 0.910121028015  0.91067553884  0.91002459135 0.910084864265];
train = [0.992250505009 0.992250505009 0.992250505009  0.992250505009 0.992250505009 0.992250505009];
figure()
hold on
size = 14
plot(n,val,'r-o','DisplayName','Validation','MarkerSize',size);
plot(n,train,'b-*','DisplayName','Training','MarkerSize',size);
xlabel('Number of trees')
ylabel('accuracy')
title('Optimizing no. of trees in a random forest')
xticks(n);
legend('show')
set(gca,'fontsize',20);