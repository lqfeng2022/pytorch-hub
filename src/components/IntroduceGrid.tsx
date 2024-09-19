import { Box, Text, Stack } from '@chakra-ui/react'
import Section from '../entities/Section-temp'

const IntroduceGrid = ({ section }: { section: Section }) => {
  return (
    <Box pt={3} pb={5}>
      <Text as='b' fontSize='lg' color='gray.600'>
        {section.name}
      </Text>
      <Stack spacing={4} py={2}>
        {section.content.map((p) => 
          <Text key={p.id}>{p.value}</Text>
        )}
      </Stack>
    </Box>
  )
} 

export default IntroduceGrid