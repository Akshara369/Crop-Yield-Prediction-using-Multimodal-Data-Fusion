from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT = Path('tmp/report_assets/dashboard_illustration.png')
W, H = 1920, 1180
BG, PANEL, LINE = '#F4F8F6', '#FFFFFF', '#D6E4DE'
GREEN, DARK, MUTED = '#20A36A', '#173C30', '#668076'
img = Image.new('RGB', (W, H), BG)
d = ImageDraw.Draw(img)
font_path = 'C:/Windows/Fonts/arial.ttf'
bold_path = 'C:/Windows/Fonts/arialbd.ttf'
def font(size, bold=False):
    return ImageFont.truetype(bold_path if bold else font_path, size)
def rounded(box, fill=PANEL, outline=LINE, radius=18, width=2):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
def txt(x, y, s, size=22, color=DARK, bold=False):
    d.text((x,y), s, font=font(size,bold), fill=color)

# App shell and persistent sidebar navigation.
rounded((12,12,W-12,H-12), fill='#F8FBF9', outline='#BBD3C8', radius=26, width=3)
rounded((30,30,322,H-30), fill='#EEF5F1', radius=20)
txt(62,70,'Crop Yield',34,DARK,True)
txt(62,118,'Smart farm planning',20,MUTED)
rounded((48,184,304,258), fill='#D9EFE4', outline='#D9EFE4', radius=14)
txt(76,207,'Dashboard',24,'#287B55',True)
txt(76,292,'Yield Forecast',23,'#47695A',True)
txt(60,H-120,'Actionable crop insights',18,MUTED)
txt(60,H-92,'for better farm planning.',18,MUTED)

# Topbar matches topbar() in app.py.
rounded((344,30,W-30,146), fill='#E4F2EB', outline='#E4F2EB', radius=20)
txt(382,54,'Crop Yield Prediction',38,DARK,True)
txt(382,105,'Actionable field insights',20,MUTED)
txt(1240,75,'Smarter Data   |   Better Insights   |   Higher Yields',22,'#587365',True)

# Home dashboard controls, KPI cards and data context strip.
txt(352,170,'Home dashboard',26,DARK,True)
labels=[('Year','2020'),('State','Maharashtra'),('Crop','Rice'),('Analyze','Yield'),('Trend','8 years')]
gap=12; start=350; total=1540; cw=(total-gap*4)//5
for i,(lab,val) in enumerate(labels):
    x=start+i*(cw+gap); rounded((x,210,x+cw,286),radius=12)
    txt(x+16,222,lab,16,MUTED); txt(x+16,250,val,20,DARK,True)
txt(356,294,'Crop selector includes Rice, Maize, Moong (Green Gram) and Urad.',16,MUTED)

metrics=[('Historical Mean Yield','From filtered records','t/ha'),('Total Area (selected)','From filtered records','ha'),('Seasonal Rainfall','From filtered records','mm'),('Soil Health (avg)','From state profile','score')]
mgap=14; mw=(1540-mgap*3)//4
for i,(lab,val,unit) in enumerate(metrics):
    x=350+i*(mw+mgap); rounded((x,326,x+mw,418),radius=15)
    d.ellipse((x+18,352,x+54,388),fill=[GREEN,'#2399C9','#7168D8','#F28B2B'][i])
    txt(x+70,345,lab,17,MUTED); txt(x+70,372,val,20,DARK,True); txt(x+70,397,unit,15,MUTED)

rounded((350,430,W-30,496),fill='#F0F6F2',outline='#DCEAE3',radius=12)
txt(374,451,'Multimodal Data Used',19,DARK,True)
txt(660,452,'Satellite: NDVI / EVI / NIR',17,MUTED)
txt(1005,452,'Weather: rainfall / climate',17,MUTED)
txt(1360,452,'Soil: pH / SOC / NPK',17,MUTED)
txt(1650,452,'Crop: area / season',17,MUTED)

# Two-column content order follows home_dashboard(): map then top states on left;
# trend then crop-area distribution and forecasting navigation on right.
left_x, gap = 350, 24
colw = (W - 30 - left_x - gap) // 2
right_x = left_x + colw + gap
def panel(x,y,w,h,title):
    rounded((x,y,x+w,y+h),radius=16)
    txt(x+24,y+18,title,22,DARK,True)
    # Neutral chart placeholder marks indicate chart location without inventing observations.
    px,py=x+54,y+68; pw,ph=w-90,h-112
    d.line((px,py+ph,px+pw,py+ph),fill='#BDCDC5',width=2)
    d.line((px,py,px,py+ph),fill='#BDCDC5',width=2)
    for k in range(1,4):
        yy=py+k*ph//4
        d.line((px,yy,px+pw,yy),fill='#E5ECE8',width=1)
    txt(px+10,py+ph+10,'Chart values are calculated from the current filters',14,MUTED)

panel(left_x,514,colw,280,'Geospatial Distribution - selected metric')
panel(right_x,514,colw,280,'Selected Metric Trend Over Time')
panel(left_x,808,colw,250,'Top States by selected metric')
panel(right_x,808,colw,250,'Crop Area Distribution')
rounded((right_x,1070,colw+right_x,H-30),fill='#F0F6F2',outline='#DCEAE3',radius=14)
txt(right_x+22,1085,'AI Yield Forecasting',20,DARK,True)
txt(right_x+250,1088,'Open Yield Forecast to enter scenario inputs and request an estimate.',16,MUTED)

img.save(OUT, quality=95)
print(OUT)
