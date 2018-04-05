% 2 dices PDF
experiment_size = 1000000;

dice1 = randi(6,experiment_size,1);
dice2 = randi(6,experiment_size,1);

throw_2_dices = arrayfun(@(x1, x2) x1+x2, dice1, dice2);

tabulate(throw_2_dices);
tbl = tabulate(throw_2_dices);

%plot(tbl(:,1), tbl(:,2), 'LineWidth', 2)
histogram(throw_2_dices)
histfit(throw_2_dices, 11, 'Normal')

% 2 dices probability distribution
function y = two_dice_throw(x)
x = round(x);
y = 0;
i = 2;

while (i>=2 && i<=7) && y == 0
    if x == i || x == 14-i
        y = (i-1)/36;
    end
    i = i+1;
end
end