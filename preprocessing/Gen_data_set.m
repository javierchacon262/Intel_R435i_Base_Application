clear all
PathR='Datos_Caving_2';
% PathB='FL_A5pw_638g'; %2x2
% PathB='FL_A5pw_315g'; %5x3
% PathB='FL_A5Pw_AGO20_714g'; %2x2
% PathB='FL_ASPw_13120_14297_JUL_18_19_54BAGS_590g'; %5x3
% PathB='FL_ASPw_16683_16925_SEP4_22BAGS_570g'; %5x3
PathB='FL_ASPw_SEP5_12_48BAGS_605g'; %5x3

PathData=[PathR '/' PathB '_Data/'];
PathPcs=[PathR '/' PathB '_Pics/'];
ds = datastore(PathData,'FileExtensions', '.xlsx','Type', 'image');
ds=ds.Files;
c=1;
Cx=5 %5 columnas
Cy=3 % 3 filas
for i=1:size(ds,1)
    N=split(ds{i},'\');
    N=split(N{end},'.');
    N=N{1};
    dsP = datastore([PathPcs N '/'],'FileExtensions', '.mat','Type', 'image');
    dsP=dsP.Files;
    for k=1:size(dsP,1)
        load(dsP{k})
        %% Input
        depth_raw=imresize(double(depth_raw),2.1);
        depth_rawR=depth_raw(208:1287,298:2217);
        color_raw=double(color_raw);
        Dato=cat(3,color_raw,depth_rawR);
        % para FL_ASPw_13120_14297_JUL_18_19_54BAGS_590g
            A=color_raw*0;
            A(100:end-150,100:end-150,:)=1;
            color_raw=color_raw+255*~A;
        %% output
        Pesos=leer_Excel(ds{i},1);
        Pesos=Pesos(1:Cy,1:Cx);
        
        RR=bwlabel(sum(color_raw,3)<200);%200
        s = regionprops(RR,'centroid','Area','FilledImage','BoundingBox');
        [V,I]=sort([s.Area],'descend');
        
        PoCu=zeros(Cy,Cx);
        Band=1;
        for z=1:Cx*Cy
            CC(z,1:2)=s(I(z)).Centroid;
            CC(z,3)=I(z);
            if s(I(z)).Area<150
                Band=0;
            end
        end
        if Band==1
            [V2,I2]=sort(CC(:,2),'ascend');
            %% agrupados cada 5
            CCa(:,1)=CC(I2,1);
            CCa(:,2)=CC(I2,2);
            CCa(:,3)=CC(I2,3);

            %% organizando dentro de esos 5
            for aa=1:Cy
                [Va,Ia]=sort(CCa(1+Cx*(aa-1):Cx*aa,1),'ascend');
                CCa(1+Cx*(aa-1):Cx*aa,1)=CCa(Ia+Cx*(aa-1),1);
                CCa(1+Cx*(aa-1):Cx*aa,2)=CCa(Ia+Cx*(aa-1),2);
                CCa(1+Cx*(aa-1):Cx*aa,3)=CCa(Ia+Cx*(aa-1),3);
            end
%                 [Va,Ia]=sort(CCa(Cx+1:Cx*2,1),'ascend');
%                 CCa(Cx+1:Cx*2,1)=CCa(Ia+Cx,1);
%                 CCa(Cx+1:Cx*2,2)=CCa(Ia+Cx,2);
%                 CCa(Cx+1:Cx*2,3)=CCa(Ia+Cx,3);
% % % 
%                 [Va,Ia]=sort(CCa(Cx*2+1:Cx*3,1),'ascend');
%                 CCa(Cx*2+1:Cx*3,1)=CCa(Ia+Cx*2,1);
%                 CCa(Cx*2+1:Cx*3,2)=CCa(Ia+Cx*2,2);
%                 CCa(Cx*2+1:Cx*3,3)=CCa(Ia+Cx*2,3);
                
            for zi=1:Cy
                for zj=1:Cx
                    PoCu(zi,zj)=CCa((zi-1)*Cx+zj,3);
                end
            end
            
            %% ahora sólo es usar PoCu y s
% %             imshow(color_raw./max(color_raw(:)))
% %             xx=3;
% %             yy=1;
% %             BB=round(s(PoCu(xx,yy)).BoundingBox);
% %             figure(2),imshow(color_raw(BB(2):BB(2)+BB(4),BB(1):BB(1)+BB(3),:)./max(color_raw(:)))
            RP=depth_rawR*0;
            for xx=1:Cy%3
                for yy=1:Cx%5
                    BB=round(s(PoCu(xx,yy)).BoundingBox);
                    AreaF=s(PoCu(xx,yy)).FilledImage;
                    PexPx=AreaF*Pesos(xx,yy)/sum(AreaF(:));
                    RP(BB(2):BB(2)+BB(4)-1,BB(1):BB(1)+BB(3)-1)=PexPx;
                end
            end 
            mkdir(['Data_train/' PathB '/'])
            save(['Data_train/' PathB '/Data_' num2str(c)],'Dato','RP','-v7')
            c=c+1;
            sum(Pesos(:))
        end
    end
end


