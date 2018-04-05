% 1 dice  PDF
x = -1:8;

x_pdf = arrayfun(@(x) single_dice_throw(x), x)

%plot(x, x_pdf, 'LineWidth', 2);
stem(x, x_pdf)

% dice probability distribution
function y = single_dice_throw(x)
x = round(x);
if x >= 1 && x <= 6
    y = 1/6;
else
    y = 0;
end
end