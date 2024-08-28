import { defineStyle, defineStyleConfig } from '@chakra-ui/react'

const thick = defineStyle({
  borderWidth: '1px', // change the width of the border
  borderStyle: 'solid', // change the style of the border
  borderColor: "orange.300",
})

const brandPrimary = defineStyle({
  borderWidth: '5px',
  borderStyle: 'dashed',
  borderColor: 'orange.500',
})

export const dividerTheme = defineStyleConfig({
  variants: { thick, brand: brandPrimary },
})