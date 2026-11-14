function temp_num_Pixelnum = generate_temp_num_Pixelnum(min_x,max_x)
    % 定义取值范围：100 到 500，步长 50
    values = min_x:50:max_x;
    
    % 生成所有 (x, y) 组合的网格
    [X, Y] = meshgrid(values, values);
    
    % 将网格展开为两列矩阵，每行对应一个 [x, y]
    temp_num_Pixelnum = [X(:), Y(:)];
end