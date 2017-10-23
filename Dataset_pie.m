close all;
X = [9870 99004 48969 35932 9786];
labels = {'0:Slovak','1:French','2:Spanish','3:German','4:Polish'};
p = pie(X)
[rows columns]=size(X)
for i = 2:2:2*columns
    t = p(i);
    t.FontSize = 18;
end

lgd = legend(labels,'Location','southoutside','Orientation','horizontal')
lgd.FontSize = 20;
saveas(gcf,'DatasetPie.png')