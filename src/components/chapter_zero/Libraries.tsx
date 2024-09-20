import { Box, Stack, Text, Image } from '@chakra-ui/react'
import { libs_img } from '../../assets/chapter_zero'
import chapterZero from '../../data/chapterZero'

const Libraries = () => {
  const libs = chapterZero[4].sections
  
  return (
    <Box pt={10}>
      <Text as='b' fontSize='lg' color='gray.600'>5. FRAMEWORKs and LIBRARIES</Text>
      <Image py={3} src={libs_img}/>
      <Stack spacing={3}>
        {libs.map((p) => 
          <div key={p.id}>
            <Text as='b'>{p.name}</Text>
            {p.content.map((t) => <Text py={1} key={t.id}>{t.value}</Text>) }
          </div>
        )}
      </Stack>
    </Box>
  )
}

export default Libraries