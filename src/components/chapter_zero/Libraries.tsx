import { Box, Stack, Text, Image } from '@chakra-ui/react'
import libraries from '../../assets/chapter_zero/libraries.jpeg'
import chapterOne from '../../data/chapterOne'

const Libraries = () => {
  const libs = chapterOne[4].sections
  return (
    <Box py={3}>
      <Text as='b' fontSize='xl' color='gray.600'>5. FRAMEWORKs and LIBRARIES</Text>
      <Image py={3} src={libraries}/>
      <Stack spacing={4}>
        {libs.map((p) => 
          <div key={p.id}>
            <Text as='b'>{p.name}</Text>
            {p.content.map((t) => <Text py={1}>{t.value}</Text>) }
          </div>
        )}
      </Stack>
    </Box>
  )
}

export default Libraries