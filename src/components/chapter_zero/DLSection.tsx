import { Box, Text } from '@chakra-ui/react'
import Definition from '../Definition'
import BaseGrid from '../BaseGrid'
import chapterZero from '../../data/chapterZero'

const DLSection = () => {
  const [
    definition, comparing, mlmodel, dlmodel, mldl, dlapps
  ] = chapterZero[2].sections
  
  return (
    <>
      <Box py={5}>
        <Text as='b' fontSize='xl' color='gray.600'>3. Deep Learning</Text>
        <Definition definition={definition}/>
      </Box>
      <BaseGrid section={comparing}/>
      <Box py={5}>
        <Text as='b' fontSize='lg' color='gray.600'>3.2 Deep Learing V.S. Machine Learing</Text>
        <BaseGrid section={mlmodel}/>
        <BaseGrid section={dlmodel}/>
        <BaseGrid section={mldl}/>
      </Box>
      <BaseGrid section={dlapps}/>
    </>
  )
}

export default DLSection