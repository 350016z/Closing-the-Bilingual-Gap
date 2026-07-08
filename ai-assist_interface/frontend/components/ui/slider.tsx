'use client';

import * as React from 'react';
import SliderBase from '@mui/material/Slider';
import Box from '@mui/material/Box';


type ScoreSliderProps = {
  value: number;
  onChange: (value: number) => void;
};

const marks = [
  {
    value: 0,
    label: (
      <div style={{ width: 100, textAlign: 'left', position: 'relative', left: '45px' }}>
        0：內容無意義<br />完全失去原意
      </div>
    ),
  },
  { value: 1, label: 1, },
  {
    value: 2,
    label: (
      <div style={{ textAlign: 'center', width: 100 }}>
        2：僅保留部分<br />原文意思
      </div>
    ),
  },
  { value: 3, label: 3, },
  {
    value: 4,
    label: (
      <div style={{ textAlign: 'center', width: 100 }}>
        4：大部分保留<br />僅少量錯誤
      </div>
    ),
  },
  { value: 5, label: 5, },
  {
    value: 6,
    label: (
      <div style={{ width: 100, textAlign: 'right', position: 'relative', right: '45px' }}>
        6：意思完全正確<br />語法也正確
      </div>
    ),
  },
];

function Slider({ value, onChange }: ScoreSliderProps) {
  return (
    <Box sx={{ width: '100%' }}>
      <SliderBase
        value={value}
        min={0}
        max={6}
        step={0.05}
        marks={marks}
        valueLabelDisplay="auto"
        onChange={(e, val) => onChange(val as number)}
        sx={{
          color: '#186cf3ff', //滑桿顏色
          height: 6,
          mt: 2,
          '& .MuiSlider-thumb': {
            width: 18,
            height: 18,
            '&:hover, &.Mui-focusVisible, &.Mui-active': {
              boxShadow: '0px 0px 0px 8px rgba(153, 162, 243, 0.64)', //滑桿圓點陰影
            },
          },
          '& .MuiSlider-track': {
            border: 'none',
          },
          '& .MuiSlider-rail': {
            opacity: 0.3,
          },
          '& .MuiSlider-markLabel': {
            fontSize: '0.75rem',
            color: '#ccc',
            mt: 1,
          },
          '& .MuiSlider-valueLabel': {
            backgroundColor: '#186cf3ff', //數值背景色
            color: '#FFFFFF', //白色
          },
        }}
      />
    </Box>
  );
}

export { Slider }