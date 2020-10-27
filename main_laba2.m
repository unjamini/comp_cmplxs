X = [ infsup(-5,5) ; infsup(-5,5) ];
[Z, WL] = globopt0(X)
color = [ 'b' 'g' 'r' 'c' 'm' 'y'];
% close all
% for i = 1:201
%     plotintval(WL(i).Box, 'n')
%     hold on
% end

vert_diff = []
hor_diff = []
x = 1:201
for i = 1:201
    vert_diff = [vert_diff, WL(i).Box(1).sup - WL(i).Box(1).inf]
    hor_diff = [hor_diff, WL(i).Box(2).sup - WL(i).Box(2).inf]
end

% axis equal
hold off
hold on;
semilogx(x, hor_diff, 'r');
hold on;
semilogx(x, vert_diff, 'c')
hold off

