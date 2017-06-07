function output =  runPatchWNet(net, input, gpu, rf, nMode)

if nMode == 0 %4
    v = ceil(size(input, 1)/2);
    h = ceil(size(input, 2)/2);
    imTL = input(1:v+rf,   1:h+rf, :);
    imBL = input(v-rf+1:end, 1:h+rf, :); 
    imTR = input(1:v+rf,   h-rf+1:end, :);
    imBR = input(v-rf+1:end, h-rf+1:end, :);

    if gpu, 
        imTL = gpuArray(imTL); 
    end;
    res    = vl_simplenn(net,imTL,[],[],1,'conserveMemory',true,'mode','test');                     
    output_T = imTL - res(end).x;
    impredTL = gather(output_T);  
    impredTL = impredTL(1:v, 1:h, :);

    if gpu, 
        imBL = gpuArray(imBL); 
    end;
    res    = vl_simplenn(net,imBL,[],[],1,'conserveMemory',true,'mode','test');                     
    output_T = imBL - res(end).x;
    impredBL = gather(output_T);  
    impredBL = impredBL(rf+1:end, 1:h, :);

    if gpu, 
        imTR = gpuArray(imTR); 
    end;
    res    = vl_simplenn(net,imTR,[],[],1,'conserveMemory',true,'mode','test');                     
    output_T = imTR - res(end).x;
    impredTR = gather(output_T);  
    impredTR = impredTR(1:v, rf+1:end, :);

    if gpu, 
        imBR = gpuArray(imBR); 
    end;
    res    = vl_simplenn(net,imBR,[],[],1,'conserveMemory',true,'mode','test');                     
    output_T = imBR - res(end).x;
    impredBR = gather(output_T);  
    impredBR = impredBR(rf+1:end, rf+1:end, :);

    impredL = cat(1, impredTL, impredBL);
    impredR = cat(1, impredTR, impredBR);
    output = cat(2, impredL, impredR);

elseif nMode ==1    %6
    if size(input, 1) > size(input, 2)
       pmode = 0;
       v = ceil(size(input, 1)/3);
       h = ceil(size(input, 2)/2); 
       imTL = input(1:v+rf,   1:h+rf, :);
       imM1 = input(v-rf+1:v*2+rf,   1:h+rf, :);
       imBL = input(size(input, 1)-v-rf+1:end, 1:h+rf, :); 
       
       imTR = input(1:v+rf,   h-rf+1:end, :);
       imM2 = input(v-rf+1:v*2+rf, h-rf+1:end, :);
       imBR = input(size(input, 1)-v-rf+1:end, h-rf+1:end, :);
    else
       pmode = 1;
       v = ceil(size(input, 1)/2);
       h = ceil(size(input, 2)/3);        
       imTL = input(1:v+rf,   1:h+rf, :);   
       imM1 = input(1:v+rf,   h-rf+1:h*2+rf, :);
       imTR = input(1:v+rf,   size(input, 2)-h-rf+1:end, :);
       
       imBL = input(v-rf+1:end, 1:h+rf, :);        
       imM2 = input(v-rf+1:end, h-rf+1:h*2+rf, :);   
       imBR = input(v-rf+1:end, size(input, 2)-h-rf+1:end, :);       
       
    end        

    if gpu, 
        imTL = gpuArray(imTL); 
    end;
    res    = vl_simplenn(net,imTL,[],[],1,'conserveMemory',true,'mode','test');                     
    output_T = imTL - res(end).x;
    impredTL = gather(output_T);  
    impredTL = impredTL(1:v, 1:h, :);

    if gpu, 
        imBL = gpuArray(imBL); 
    end;
    res    = vl_simplenn(net,imBL,[],[],1,'conserveMemory',true,'mode','test');                     
    output_T = imBL - res(end).x;
    impredBL = gather(output_T);  
    impredBL = impredBL(rf+1:end, 1:h, :);

    if gpu, 
        imTR = gpuArray(imTR); 
    end;
    res    = vl_simplenn(net,imTR,[],[],1,'conserveMemory',true,'mode','test');                     
    output_T = imTR - res(end).x;
    impredTR = gather(output_T);  
    impredTR = impredTR(1:v, rf+1:end, :);

    if gpu, 
        imBR = gpuArray(imBR); 
    end;
    res    = vl_simplenn(net,imBR,[],[],1,'conserveMemory',true,'mode','test');                     
    output_T = imBR - res(end).x;
    impredBR = gather(output_T);  
    impredBR = impredBR(rf+1:end, rf+1:end, :);       
    
    if pmode == 0         
        if gpu, 
          imM1 = gpuArray(imM1); 
        end;
        res    = vl_simplenn(net,imM1,[],[],1,'conserveMemory',true,'mode','test');                     
        output_T = imM1 - res(end).x;
        impredM1 = gather(output_T);                
        impredM1 = impredM1(rf+1:end-rf,1:h,:);

        if gpu, 
              imM2 = gpuArray(imM2); 
        end;
        res    = vl_simplenn(net,imM2,[],[],1,'conserveMemory',true,'mode','test');                     
        output_T = imM2 - res(end).x;
        impredM2 = gather(output_T);                
        impredM2 = impredM2(rf+1:end-rf, rf+1:end, :); 
        
        impredL = cat(1, impredTL, impredM1);
        impredL = cat(1, impredL, impredBL);
        impredR = cat(1, impredTR, impredM2);
        impredR = cat(1, impredR, impredBR);
        output = cat(2, impredL, impredR);
    else
        if gpu, 
          imM1 = gpuArray(imM1); 
        end;
        res    = vl_simplenn(net,imM1,[],[],1,'conserveMemory',true,'mode','test');                     
        output_T = imM1 - res(end).x;
        impredM1 = gather(output_T);                
        impredM1 = impredM1(1:v, rf+1:end-rf, :);

        if gpu, 
              imM2 = gpuArray(imM2); 
        end;
        res    = vl_simplenn(net,imM2,[],[],1,'conserveMemory',true,'mode','test');                     
        output_T = imM2 - res(end).x;
        impredM2 = gather(output_T);                
        impredM2 = impredM2(rf+1:end, rf+1:end-rf, :); 
        
        impredL = cat(2, impredTL, impredM1);
        impredL = cat(2, impredL, impredTR);
        impredR = cat(2, impredBL, impredM2);
        impredR = cat(2, impredR, impredBR);
        output = cat(1, impredL, impredR);        
    end    

else % mode2 %8
    v = ceil(size(input, 1)/3);
    h = ceil(size(input, 2)/3); 
        
    imTL = input(1:v+rf,   1:h+rf, :);
    imM1 = input(v-rf+1:v*2+rf,   1:h+rf, :);
    imBL = input(size(input, 1)-v-rf+1:end, 1:h+rf, :); 

    imTR = input(1:v+rf,   size(input, 2)-h-rf+1:end, :);
    imM2 = input(v-rf+1:v*2+rf, size(input, 2)-h-rf+1:end, :);
    imBR = input(size(input, 1)-v-rf+1:end, size(input, 2)-h-rf+1:end, :);
   
    imTM = input(1:v+rf,   h-rf+1:h*2+rf, :);   
    imM3 = input(v-rf+1:v*2+rf,   h-rf+1:h*2+rf, :);
    imBM = input(size(input, 1)-v-rf+1:end, h-rf+1:h*2+rf, :);                   
       
    if gpu, 
        imTL = gpuArray(imTL); 
    end;
    res    = vl_simplenn(net,imTL,[],[],1,'conserveMemory',true,'mode','test');                     
    output_T = imTL - res(end).x;
    impredTL = gather(output_T);  
    impredTL = impredTL(1:v, 1:h, :);

    if gpu, 
        imBL = gpuArray(imBL); 
    end;
    res    = vl_simplenn(net,imBL,[],[],1,'conserveMemory',true,'mode','test');                     
    output_T = imBL - res(end).x;
    impredBL = gather(output_T);  
    impredBL = impredBL(rf+1:end, 1:h, :);

    if gpu, 
        imTR = gpuArray(imTR); 
    end;
    res    = vl_simplenn(net,imTR,[],[],1,'conserveMemory',true,'mode','test');                     
    output_T = imTR - res(end).x;
    impredTR = gather(output_T);  
    impredTR = impredTR(1:v, rf+1:end, :);

    if gpu, 
        imBR = gpuArray(imBR); 
    end;
    res    = vl_simplenn(net,imBR,[],[],1,'conserveMemory',true,'mode','test');                     
    output_T = imBR - res(end).x;
    impredBR = gather(output_T);  
    impredBR = impredBR(rf+1:end, rf+1:end, :);       
    
    if gpu, 
          imM1 = gpuArray(imM1); 
    end;
    res    = vl_simplenn(net,imM1,[],[],1,'conserveMemory',true,'mode','test');                     
    output_T = imM1 - res(end).x;
    impredM1 = gather(output_T);                
    impredM1 = impredM1(rf+1:end-rf,1:h,:);

    if gpu, 
          imM2 = gpuArray(imM2); 
    end;
    res    = vl_simplenn(net,imM2,[],[],1,'conserveMemory',true,'mode','test');                     
    output_T = imM2 - res(end).x;
    impredM2 = gather(output_T);                
    impredM2 = impredM2(rf+1:end-rf, rf+1:end, :); 

    if gpu, 
        imTM = gpuArray(imTM); 
    end;
    res    = vl_simplenn(net,imTM,[],[],1,'conserveMemory',true,'mode','test');                     
    output_T = imTM - res(end).x;
    impredTM = gather(output_T);  
    impredTM = impredTM(1:v, rf+1:end-rf, :);
    
    if gpu, 
        imM3 = gpuArray(imM3); 
    end;
    res    = vl_simplenn(net,imM3,[],[],1,'conserveMemory',true,'mode','test');                     
    output_T = imM3 - res(end).x;
    impredM3 = gather(output_T);                
    impredM3 = impredM3(rf+1:end-rf, rf+1:end-rf, :); 
    
    if gpu, 
        imBM = gpuArray(imBM); 
    end;
    res    = vl_simplenn(net,imBM,[],[],1,'conserveMemory',true,'mode','test');                     
    output_T = imBM - res(end).x;
    impredBM = gather(output_T);                
    impredBM = impredBM(rf+1:end, rf+1:end-rf, :); 
            
    impredL = cat(1, impredTL, impredM1);
    impredL = cat(1, impredL, impredBL);
    impredR = cat(1, impredTR, impredM2);
    impredR = cat(1, impredR, impredBR);
    impredM = cat(1, impredTM, impredM3);
    impredM = cat(1, impredM, impredBM);
    output = cat(2, impredL, impredM);
    output = cat(2, output, impredR);
end