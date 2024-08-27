import { Box, Stack, Text, Image, HStack } from '@chakra-ui/react'
import prere_img from '../../assets/chapter_zero/prerequisites.jpeg'
import chapterZero from '../../data/chapterZero'

const Prerequisites = () => {
  const preres = chapterZero[6].sections
  return (
    <Box py={5}>
      <Text as='b' fontSize='xl' color='gray.600'>7. Prerequisites</Text>
      <Image py={3} src={prere_img}/>
      <Stack spacing={4}>
        {preres.map((p) => 
          <Stack key={p.id}>
            <Text as='b'>{p.name}</Text>
            {p.content.map((t) => 
              <div key={t.id}>
                {t.title && <HStack>
                  <Text>{t.id}</Text>
                  <Text as='b'>{t.title}</Text>
                </HStack>}
                <Text py={1}>{t.value}</Text>
              </div>
            )}
          </Stack>
        )}
      </Stack>
    </Box>
  )
}

export default Prerequisites